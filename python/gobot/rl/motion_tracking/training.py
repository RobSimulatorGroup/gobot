"""Checkpointed SuperMimic training over the scene-owned tracking environment."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable
import json
import hashlib
import torch

from gobot.rl.experiment_identity import canonical_digest
from gobot.rl.policy import POLICY_MANIFEST_KEY, PolicyManifest, policy_manifest_from_checkpoint, write_policy_manifest_sidecar
from gobot.rl.supermimic import SuperMimicConfig, SuperMimicLearner
from .manifest import build_tracking_manifest


class TrainingStop(Exception):
    """A controlled stop at an iteration boundary, preserving a full checkpoint."""


def policy_input_contract(manifest: PolicyManifest):
    metadata=manifest.metadata()
    return {key:metadata[key] for key in ('observation_spec','action_spec','joint_names','control','model')}


def read_checkpoint(path):
    # Hash the same open inode that torch.load read. Atomic last.pt replacement
    # by another process cannot associate new metadata with older loaded weights.
    with Path(path).open('rb') as stream:
        payload=torch.load(stream,map_location='cpu',weights_only=True)
        stream.seek(0);digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return payload,'sha256:'+digest.hexdigest()


def training_contract(env, algorithm, *, software_digest):
    options=asdict(algorithm)
    # These settings govern duration and reporting, not a learning transition.
    for name in ('iterations','ckpt_every','log_every','log_backend'):
        options.pop(name,None)
    manifest=build_tracking_manifest(env,algorithm)
    task={key:value for key,value in asdict(env.cfg).items() if key not in ('project_path','motion_files')}
    value={'version':1,'policy':policy_input_contract(manifest),'task':task,
           'physics':env.batch.artifact.digest,'backend':env.batch.backend,'num_envs':env.num_envs,
           'data':manifest.extras['data'],'algorithm':options,'software_digest':software_digest,
           'resume_episodes':'fresh','resume_replay':'refill'}
    return value,canonical_digest(value),manifest


class TrackingTrainingSession(SuperMimicLearner):
    """Own optimizer/task/RNG restoration and explicit physics-transfer phases.

    A fork begins with fresh optimizer, critic and replay state. It preserves the
    actor and its input statistics and keeps that actor frozen until a separately
    validated promotion. Resume keeps the current phase and starts fresh episodes.
    """

    def __init__(self, env, cfg: SuperMimicConfig, log_dir, *, software_digest,
                 should_stop: Callable[[], bool] | None = None, progress=None):
        self.contract,self.contract_digest,self.manifest=training_contract(env,cfg,software_digest=software_digest)
        self.phase='fresh'
        self.lineage=[]
        self.validation_history=[]
        self._should_stop=should_stop
        self._progress=progress
        self._loaded_checkpoint=None
        self._loaded_checkpoint_digest=None
        super().__init__(env,cfg,log_dir,backend=env.batch.backend)

    def _checkpoint(self, include_optimizers):
        payload=super()._checkpoint(include_optimizers)
        payload[POLICY_MANIFEST_KEY]=self.manifest.metadata()
        payload['gobot_training']={
            'version':1,'contract':self.contract,'contract_digest':self.contract_digest,
            'phase':self.phase,'lineage':self.lineage,'validation_history':self.validation_history,
            'task_state':self.env.training_state_dict(),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None,
        }
        return payload

    def save(self, path=None, include_optimizers=True):
        destination=super().save(path,include_optimizers)
        write_policy_manifest_sidecar(destination,self.manifest)
        return destination

    def collect(self, n_steps):
        if self._should_stop and self._should_stop():
            raise TrainingStop('stop requested')
        super().collect(n_steps)
        if self._progress:
            self._progress(self.iteration,self.sim_steps,self.phase)

    def fork(self, checkpoint, *, source_manifest=None, source_identity=None, load_world_model=True):
        if self.iteration or self.buffer.size:
            raise ValueError('fork requires a fresh training session')
        payload,source_digest=read_checkpoint(checkpoint)
        manifest=policy_manifest_from_checkpoint(payload)
        if manifest is None:
            if source_manifest is None:
                raise ValueError('legacy checkpoint import requires an explicit source policy manifest')
            manifest=source_manifest if isinstance(source_manifest,PolicyManifest) else PolicyManifest.from_json(Path(source_manifest).read_text())
            binding=manifest.extras.get('checkpoint_sha256')
            if source_identity is not None:
                identity=source_identity if isinstance(source_identity,dict) else json.loads(Path(source_identity).read_text())
                binding=identity.get('checkpoint_sha256') or (identity.get('parent') or {}).get('sha256')
            if not binding or binding.removeprefix('sha256:')!=source_digest.removeprefix('sha256:'):
                raise ValueError('legacy manifest requires a matching checkpoint hash or source identity')
        if canonical_digest(policy_input_contract(manifest))!=canonical_digest(policy_input_contract(self.manifest)):
            raise ValueError('fork policy inputs/control differ; use explicit conversion or distillation')
        source_split=manifest.extras.get('data',{}).get('split')
        if self.env.cfg.split=='train' and source_split!='train':
            raise ValueError('strict subject-split training cannot inherit all-library/unknown teacher weights')
        old=payload.get('gobot_training',{})
        ancestors=old.get('lineage',[])
        if self.env.cfg.split=='train' and any(item.get('data_split')!='train' for item in ancestors):
            raise ValueError('checkpoint lineage contains held-out or unknown data')
        self.pi.load_state_dict(payload['pi'],strict=True)
        if load_world_model:self.wm.load_state_dict(payload['wm'],strict=True)
        self.lineage=[*ancestors,{'checkpoint_sha256':source_digest,'data_split':source_split,
                                 'source_contract':old.get('contract_digest'),'mode':'fork',
                                 'world_model_weights_loaded':bool(load_world_model)}]
        self.phase='adaptation'
        self.set_actor_frozen(True)
        self.env.reset()

    def load(self, path):
        payload,source_digest=read_checkpoint(path)
        saved=payload.get('gobot_training')
        if not saved or saved.get('version')!=1:
            raise ValueError('resume requires a Gobot training checkpoint; import legacy weights with fork')
        if canonical_digest(saved.get('contract'))!=saved.get('contract_digest'):
            raise ValueError('saved training contract is corrupt')
        if saved['contract_digest']!=self.contract_digest:
            raise ValueError('resume configuration/data/software/batch changed; use an explicit fork')
        if saved.get('phase') not in ('fresh','adaptation','joint'):
            raise ValueError('unknown saved training phase')
        if saved['phase']=='adaptation' and not payload.get('actor_frozen'):
            raise ValueError('adaptation checkpoint must keep the actor frozen')
        required={'critic','opt_wm','opt_pi','opt_v'}
        if not required<=payload.keys():
            raise ValueError('resume requires a full checkpoint with optimizer and critic states')
        self.env.load_training_state_dict(saved['task_state'])
        self._restore_checkpoint(payload)
        self.phase=saved['phase'];self.lineage=list(saved['lineage'])
        self.validation_history=list(saved.get('validation_history',[]))
        torch.set_rng_state(saved['torch_rng'].cpu())
        if self.device.type=='cuda':
            if saved.get('cuda_rng') is None:raise ValueError('CUDA resume checkpoint lacks generator state')
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device=self.device)
        self._loaded_checkpoint=Path(path)
        self._loaded_checkpoint_digest=source_digest

    def promote(self, validation):
        """Advance a resumed adaptation checkpoint only with matching evidence."""
        if self.phase!='adaptation' or self._loaded_checkpoint is None:
            raise ValueError('promotion requires a resumed adaptation checkpoint')
        if validation.get('protocol')!='tracking_adaptation_validation_v1' or validation.get('passed') is not True:
            raise ValueError('independent adaptation validation has not passed')
        if (validation.get('metrics_only') is not False or validation.get('thresholds') is None or
                validation.get('thresholds_digest')!=canonical_digest(validation['thresholds'])):
            raise ValueError('promotion requires recorded, consistent acceptance thresholds')
        if validation.get('checkpoint_sha256')!=self._loaded_checkpoint_digest or validation.get('contract_digest')!=self.contract_digest:
            raise ValueError('validation belongs to a different checkpoint or training contract')
        required=('world_model_1_8_32','critic','physical_rollout')
        if any(validation.get('checks',{}).get(name,{}).get('passed') is not True for name in required):
            raise ValueError('adaptation validation is incomplete')
        self.validation_history.append(validation)
        self.phase='joint';self.set_actor_frozen(False)

    def train(self, resume=None):
        if resume is not None:self.load(resume)
        # Always leave a usable initial checkpoint if compilation/preemption
        # interrupts the first update. The replay intentionally is not serialized.
        self.save()
        try:
            return super().train()
        except TrainingStop:
            self.save()
            return {'completed':False,'iteration':self.iteration,'sim_steps':self.sim_steps,
                    'phase':self.phase,'stop_reason':'controlled_stop'}
