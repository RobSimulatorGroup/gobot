"""XY​​ZW motion features and the reference clip interpolation convention."""
import torch
from gobot.rl.supermimic.rotation_primitives import matrix_from_quat


def quat_to_matrix(q):
    return matrix_from_quat(q[..., (3,0,1,2)])


def axis_angle_to_quat(axis, angle):
    half = angle[..., None]/2
    value = torch.cat((axis/axis.norm(dim=-1,keepdim=True)*half.sin(), half.cos()), -1)
    return value/value.norm(dim=-1,keepdim=True)


def quat_to_tan_norm(q):
    def rotate(v):
        cross = 2*torch.linalg.cross(q[..., :3],v)
        return v+q[...,3:4]*cross+torch.linalg.cross(q[...,:3],cross)
    x=torch.zeros_like(q[...,:3]);x[...,0]=1
    z=torch.zeros_like(x);z[...,2]=1
    return torch.cat((rotate(x),rotate(z)),-1)


def quat_twist_angle(q, axis):
    vector = (q[...,:3]*axis).sum(-1,keepdim=True)*axis
    twist = torch.cat((vector,q[...,3:4]),-1)
    twist = twist/twist.norm(dim=-1,keepdim=True)
    twist = torch.where(twist[...,3:4]<0,-twist,twist)
    length = twist[...,:3].norm(dim=-1)
    angle = 2*torch.atan2(length,twist[...,3])
    angle = torch.where((twist[...,:3]*axis).sum(-1)<0,-angle,angle)
    return torch.where(length>1.e-5,angle,torch.zeros_like(angle))


def slerp(q0,q1,t):
    # Adapted from MimicKit fbe1a36, util/torch_util.py (Apache-2.0).
    # Preserve its small-angle midpoint fallback, including at clip boundaries.
    # See LICENSE-MimicKit and NOTICE.md beside this module.
    cosine=(q0*q1).sum(-1)
    q1=torch.where((cosine<0)[...,None],-q1,q1)
    cosine=cosine.abs()[...,None]
    theta=torch.acos(cosine)
    sine=torch.sqrt(1-cosine.square())
    t=t[...,None]
    value=torch.sin((1-t)*theta)/sine*q0+torch.sin(t*theta)/sine*q1
    value=torch.where(sine.abs()<.001,.5*q0+.5*q1,value)
    return torch.where(cosine.abs()>=1,q0,value)
