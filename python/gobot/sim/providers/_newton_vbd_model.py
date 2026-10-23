"""Compile Gobot physics snapshots into Newton data; no SceneTree traversal."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def matrix(transform):
    value = np.asarray(transform["matrix"], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(value).all():
        raise ValueError("physics transforms must be finite")
    return value


def xform(wp, value):
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError("Newton rigid transforms require unit scale and a proper rotation")
    return wp.transform(value[:3, 3], Rotation.from_matrix(rotation).as_quat())


def interacts(a, b):
    return bool((a["collision_layer"] & b["collision_mask"]) or
                (b["collision_layer"] & a["collision_mask"]))


class ModelCompiler:
    def __init__(self, snapshot, config, newton, wp):
        self.snapshot, self.config, self.newton, self.wp = snapshot, config, newton, wp
        self.builder = newton.ModelBuilder(gravity=tuple(snapshot["gravity"]))
        self.links, self.joints, self.shapes, self.soft_bodies = {}, {}, [], []
        self.body_names, self.particle_owners = [], []
        self.initial_joint_positions = {}
        self.meshes = {}

    def compile(self):
        b, wp = self.builder, self.wp
        for name in ("coupling_count", "attachment_count", "sensor_count"):
            if self.snapshot.get(name, 0):
                raise ValueError(f"Newton VBD does not yet implement snapshot {name}")
        for robot in self.snapshot["robots"]:
            self._robot(robot)
        for shape in self.snapshot["loose_shapes"]:
            self._shape(shape, -1, np.eye(4), ("", shape["name"]))
        for terrain in self.snapshot["terrains"]:
            if terrain["heightfields"]:
                raise ValueError("Newton VBD snapshot heightfields are not supported yet")
            for i, geometry in enumerate(terrain["boxes"] + terrain["meshes"]):
                shape = dict(terrain, **geometry, type=1 if "size" in geometry else 5,
                             disabled=False, name=f'{terrain["name"]}/{i}')
                self._shape(shape, -1, np.eye(4), (terrain["name"], shape["name"]))
        for body in self.snapshot["deformables"]:
            self._soft(body)
        for i, (a, _) in enumerate(self.shapes):
            for j in range(i):
                if not interacts(a, self.shapes[j][0]):
                    b.add_shape_collision_filter_pair(i, j)
        # A hinge couples all four vertices, including the two opposite ones
        # that do not share a triangle. They must not update in the same VBD
        # color, especially on the narrow, folded seams of a closed film.
        b.color(include_bending=True)
        self.model = b.finalize(device=self.config.device)
        if self.model.joint_count:
            # Use a consistent quaternion branch along each articulation. Equivalent
            # independently exported q/-q frames can otherwise look like a 2*pi
            # joint error to Newton's maximal-coordinate joint constraints.
            newton = self.newton
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.soft_contact_ke = self.config.soft_contact_stiffness
        self.model.soft_contact_kd = self.config.soft_contact_damping
        self.model.soft_contact_mu = self.config.soft_friction
        self.particle_owners = np.asarray(self.particle_owners, dtype=np.int32)
        return self.model

    def _robot(self, robot):
        b, wp = self.builder, self.wp
        name = robot["name"]
        links = {link["name"]: link for link in robot["links"]}
        incoming = {j["child_link"] for j in robot["joints"]}
        for link in robot["links"]:
            key = (name, link["name"])
            if key in self.links:
                raise ValueError(f"duplicate physics link {key}")
            inertia = np.diag(link["inertia_diagonal"])
            xy, xz, yz = link["inertia_off_diagonal"]
            inertia += np.array([[0, xy, xz], [xy, 0, yz], [xz, yz, 0]])
            q = link["inertia_orientation"]
            rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            inertia = rotation @ inertia @ rotation.T
            body = b.add_link(xform=xform(wp, matrix(link["global_transform"])),
                              mass=link["mass"], com=wp.vec3(*link["center_of_mass"]),
                              inertia=wp.mat33(*inertia.flatten()), lock_inertia=True,
                              is_kinematic=(link["mass"] <= 0 or
                                  (not robot["standalone_rigid_body"] and link["name"] not in incoming)),
                              label=f"{name}/{link['name']}")
            self.links[key] = body
            self.body_names.append(key)
            for shape in link["shapes"]:
                self._shape(shape, body, matrix(link["global_transform"]), key)
        articulation = []
        # Root links in fixed robots are world-fixed; standalone bodies are free.
        for link in robot["links"]:
            if link["name"] in incoming:
                continue
            body = self.links[(name, link["name"])]
            if robot["standalone_rigid_body"] and link["mass"] > 0:
                joint = b.add_joint_free(child=body, label=f"{name}/free")
            else:
                joint = b.add_joint_fixed(parent=-1, child=body,
                            parent_xform=xform(wp, matrix(link["global_transform"])),
                            label=f"{name}/{link['name']}/anchor")
            articulation.append(joint)
        pending = list(robot["joints"])
        ready = set(links) - incoming
        while pending:
            progressed = False
            for joint in pending[:]:
                parent_name, child_name = joint["parent_link"], joint["child_link"]
                if parent_name and parent_name not in ready:
                    continue
                parent = self.links[(name, parent_name)] if parent_name else -1
                child = self.links[(name, child_name)]
                frame = matrix(joint["global_transform"])
                parent_pose = matrix(links[parent_name]["global_transform"]) if parent >= 0 else np.eye(4)
                child_pose = matrix(links[child_name]["global_transform"])
                args = dict(parent=parent, child=child, parent_xform=xform(wp, np.linalg.inv(parent_pose) @ frame),
                            child_xform=xform(wp, np.linalg.inv(child_pose) @ frame),
                            label=f"{name}/{joint['name']}")
                kind = joint["type"]
                offset = float(joint["joint_position"])
                if kind == 0:
                    index = b.add_joint_fixed(**args)
                elif kind in (1, 2, 3):
                    if joint["drive_mode"] not in (0, 2):
                        raise ValueError("Newton VBD snapshot joints currently support passive and position drives")
                    lower, upper = joint["lower_limit"] - offset, joint["upper_limit"] - offset
                    if kind == 2:
                        lower, upper = -1e6, 1e6
                    fn = b.add_joint_prismatic if kind == 3 else b.add_joint_revolute
                    index = fn(**args, axis=wp.vec3(*joint["axis"]),
                               target_ke=joint["drive_stiffness"] if joint["drive_mode"] == 2 else 0.,
                               target_kd=joint["drive_damping"] if joint["drive_mode"] == 2 else joint["damping"],
                               target_pos=0., limit_lower=lower, limit_upper=upper,
                               limit_ke=self.config.joint_limit_stiffness, limit_kd=10.)
                    self.joints[(name, joint["name"])] = dict(joint, index=index,
                            q=b.joint_q_start[index], dof=b.joint_qd_start[index], offset=offset,
                            effective_mass=0.)
                else:
                    raise ValueError(f"unsupported Newton VBD joint {name}/{joint['name']} type {kind}")
                articulation.append(index)
                ready.add(child_name)
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError(f"unresolved or cyclic joints in {name}")
        if articulation:
            b.add_articulation(articulation, label=name)

    def _shape(self, shape, body, body_transform, names):
        if shape["disabled"]:
            return
        b, wp = self.builder, self.wp
        material = shape["material"]
        compliance = float(material["compliance"])
        cfg = self.newton.ModelBuilder.ShapeConfig(density=0.,
                    mu=float(material["friction"]),
                    ke=1. / compliance if compliance > 0 else self.config.contact_stiffness,
                    kd=self.config.contact_damping,
                    margin=max(0., shape["rest_offset"]), gap=max(self.config.contact_gap, shape["contact_offset"]),
                    restitution=float(material["restitution"]))
        kind = shape["type"]
        if kind != 5:
            cfg.configure_sdf(force_sdf=True, max_resolution=64)
        args = dict(body=body, xform=xform(wp, np.linalg.inv(body_transform) @ matrix(shape["transform"])),
                    cfg=cfg, label=shape["path"])
        if kind == 1:
            size = np.asarray(shape["size"]) / 2
            index = b.add_shape_box(**args, hx=size[0], hy=size[1], hz=size[2])
        elif kind == 2:
            index = b.add_shape_sphere(**args, radius=shape["radius"])
        elif kind in (3, 4):
            fn = b.add_shape_cylinder if kind == 3 else b.add_shape_capsule
            index = fn(**args, radius=shape["radius"], half_height=shape["height"] / 2)
        elif kind == 5:
            vertices = np.asarray(shape["vertices"], dtype=np.float32)
            indices = np.asarray(shape["indices"], dtype=np.int32)
            key = (vertices.tobytes(), indices.tobytes())
            if key not in self.meshes:
                mesh = self.newton.Mesh(vertices, indices)
                # Newton's mesh-backed shapes own their SDF. ShapeConfig's
                # sdf_* baking options are only accepted for primitives.
                mesh.build_sdf(device=self.config.device, max_resolution=128,
                               margin=.005, narrow_band_range=(-.02, .02),
                               cache_dir=Path(wp.config.kernel_cache_dir) / "newton_sdf")
                self.meshes[key] = mesh
            index = b.add_shape_mesh(**args, mesh=self.meshes[key])
        else:
            raise ValueError(f"unsupported shape {shape['path']} type {kind}")
        if index != len(self.shapes):
            raise RuntimeError("Newton shape layout changed during compilation")
        self.shapes.append((shape, names))

    def _soft(self, body):
        b, wp = self.builder, self.wp
        if body["kinematic"]:
            raise ValueError("kinematic deformables are not supported by Newton VBD provider")
        start = len(b.particle_q)
        transform = matrix(body["transform"])
        vertices = np.asarray(body["vertices"], dtype=np.float64)
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        young, nu = body["young_modulus"], body["poisson_ratio"]
        mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
        args = dict(pos=wp.vec3(), rot=wp.quat_identity(), scale=1., vel=wp.vec3(),
                    vertices=world.tolist(), particle_radius=self.config.particle_radius)
        if body["model"] == 1:
            thickness = body["thickness"]
            b.add_cloth_mesh(**args, indices=body["triangles"], density=body["density"] * thickness,
                            tri_ke=mu * thickness, tri_ka=lam * thickness,
                            tri_kd=mu * thickness * self.config.strain_relaxation_time,
                            edge_ke=body["bending_stiffness"],
                            edge_kd=body["bending_stiffness"] * self.config.strain_relaxation_time)
        elif body["model"] == 0:
            b.add_soft_mesh(**args, indices=body["tetrahedra"], density=body["density"],
                           k_mu=mu, k_lambda=lam, k_damp=mu * self.config.strain_relaxation_time)
        else:
            raise ValueError(f"unsupported deformable model {body['model']}")
        count = len(vertices)
        self.particle_owners.extend([len(self.soft_bodies)] * count)
        self.soft_bodies.append(dict(body, element_offset=start, element_count=count,
                                    inverse_transform=np.linalg.inv(transform)))

    def filter_contacts(self, pipeline):
        """Apply Gobot collision layers to all three Newton soft contact paths."""
        wp, model = self.wp, self.model
        if not model.particle_count:
            return
        owners = self.particle_owners
        edges = model.edge_indices.numpy()
        triangles = model.tri_indices.numpy()
        for field, elements in (("soft_rigid_contact_pairs", None),
                                ("soft_edge_rigid_pairs", edges[:, 2:]),
                                ("soft_face_rigid_pairs", triangles)):
            pairs = getattr(pipeline, field).numpy()
            if not len(pairs):
                continue
            particle = pairs[:, 0] if elements is None else elements[pairs[:, 0], 0]
            allowed = np.array([[interacts(soft, shape) for shape, _ in self.shapes]
                                for soft in self.soft_bodies], dtype=bool)
            keep = allowed[owners[particle], pairs[:, 1]]
            # Static boxes need edge/face contacts too: a cloth triangle can
            # straddle a table edge while all its vertices lie outside it.
            setattr(pipeline, field, wp.array(pairs[keep], dtype=wp.vec2i, device=model.device))
        pipeline._soft_contact_pair_count = len(pipeline.soft_rigid_contact_pairs)

    def self_contact_filters(self):
        """Filter by authored layers; keep inter-body shell/fill contact enabled."""
        owners = self.particle_owners
        if not len(owners):
            return {}, {}
        triangles, edges = self.model.tri_indices.numpy(), self.model.edge_indices.numpy()
        triangle_owner = owners[triangles[:, 0]]
        edge_owner = owners[edges[:, 2]]
        vf, ee = {}, {}
        for i, body in enumerate(self.soft_bodies):
            excluded = [j for j, other in enumerate(self.soft_bodies)
                        if (i == j and not body["self_collision_enabled"]) or
                        (i != j and not interacts(body, other))]
            tf = np.flatnonzero(np.isin(triangle_owner, excluded)).tolist()
            ef = np.flatnonzero(np.isin(edge_owner, excluded)).tolist()
            if tf:
                for vertex in np.flatnonzero(owners == i):
                    vf[int(vertex)] = tf
            if ef:
                for edge in np.flatnonzero(edge_owner == i):
                    ee[int(edge)] = ef
        return vf, ee

    def finalize_self_contact_filters(self, detector):
        """Make Newton's edge exclusions reciprocal before graph capture.

        Newton 1.6 evaluates each side of an edge-edge contact in a separate
        query. Its topological filter can exclude (a, b) but retain (b, a):
        bending adjacency includes opposite hinge vertices, which are not edge
        endpoints. Such a one-sided contact applies an internal force without
        its reaction. Take the union with the transpose, preserving every
        authored exclusion and excluding the reverse pair as well. Vertex-face
        contacts already apply all four reactions in one evaluation.
        """
        if detector is None or detector.edge_filtering_list is None:
            return
        from scipy.sparse import csr_matrix
        indices = detector.edge_filtering_list.numpy()
        offsets = detector.edge_filtering_list_offsets.numpy()
        filters = csr_matrix((np.ones(len(indices), dtype=bool), indices, offsets),
                             shape=(self.model.edge_count, self.model.edge_count))
        reciprocal = filters.maximum(filters.T)
        if reciprocal.nnz == filters.nnz:
            return
        reciprocal.sort_indices()
        wp = self.wp
        detector.edge_filtering_list = wp.array(reciprocal.indices, dtype=wp.int32,
                                                device=self.model.device)
        detector.edge_filtering_list_offsets = wp.array(reciprocal.indptr, dtype=wp.int32,
                                                        device=self.model.device)
