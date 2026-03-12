import torch
from utils.graphics_utils import sample_incident_rays
import torch.nn.functional as F
import numpy as np

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec



### USING ###
def rendering_equation_lidar(base_color, roughness, normals, viewdirs, view_dists):
    #### neglecting d^2 in rendering equation
    normals = normals.detach()
    #roughness = roughness.detach()
    
    n_d_i = (normals * viewdirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] #/ np.pi
    f_s = 0 #GGX_specular(normals, viewdirs, viewdirs[:,None, :], roughness, fresnel=0.04)

    transport =  1 #n_d_i[:,None,:]  # （num_pts, num_sample, 1)
    #specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2) / (view_dists **2)
    return pbr



### USING ###
def rendering_equation(base_color, roughness, normals, viewdirs,
                              incidents=None, direct_light_env_light=None,
                              incident_dirs=None, incident_areas=None, visibility_precompute=None,sample_num=24,sun_visibility=None,xyz=None,step=None,
                              point_chunk_size: int = 0, compute_dtype: str = "float16"):
    
    normals = normals.detach()
    # if incident_dirs is None:
    #     incident_dirs, incident_areas = sample_incident_rays(normals, True, sample_num)
    with_sun = True
    if with_sun: #else:
        sun_visibility = visibility_precompute[:,0,:]
        sun_direction = incident_dirs[:,0,:]
        incident_areas = incident_areas[:,1:,:]
        visibility_precompute = visibility_precompute[:,1:,:]
        incident_dirs = incident_dirs[:,1:,:]
    else:        
        sun_direction = direct_light_env_light.get_sun_direction()
        sun_direction = sun_direction[None,...].repeat(incident_dirs.shape[0],1)


    deg = int(np.sqrt(incidents.shape[1]) - 1)
    

    num_pts = base_color.shape[0]
    if point_chunk_size is None or point_chunk_size <= 0:
        point_chunk_size = num_pts
    work_dtype = torch.float16 if compute_dtype == "float16" else normals.dtype

    pbr_chunks = []
    diffuse_chunks = []
    for start in range(0, num_pts, point_chunk_size):
        end = min(start + point_chunk_size, num_pts)
        normals_chunk = normals[start:end].to(device=normals.device, dtype=work_dtype, non_blocking=True)
        dirs_chunk = incident_dirs[start:end].to(device=normals.device, dtype=work_dtype, non_blocking=True)
        areas_chunk = incident_areas[start:end].to(device=normals.device, dtype=work_dtype, non_blocking=True)
        vis_chunk = visibility_precompute[start:end].to(device=normals.device, dtype=work_dtype, non_blocking=True)
        global_incident_lights = torch.ones_like(dirs_chunk) * torch.tensor([200, 200, 180]).to(device=normals.device) / 255 * 1.5
        incident_lights = global_incident_lights * vis_chunk
        n_d_i_chunk = (normals_chunk[:, None] * dirs_chunk).sum(-1, keepdim=True).clamp(min=0)
        f_d_chunk = base_color[start:end, None].to(dtype=work_dtype) / np.pi
        transport_chunk = incident_lights * areas_chunk * n_d_i_chunk  # (num_pts_chunk, num_sample, 3)
        pbr_chunks.append((f_d_chunk * transport_chunk).mean(dim=-2).to(dtype=base_color.dtype))
        diffuse_chunks.append(transport_chunk.mean(dim=-2).to(dtype=base_color.dtype))

    pbr = torch.cat(pbr_chunks, dim=0)
    diffuse_light = torch.cat(diffuse_chunks, dim=0)
    specular = torch.zeros_like(pbr)

    if with_sun and (sun_visibility is not None):
        sun_visibility = sun_visibility.detach().to(device=normals.device, dtype=normals.dtype, non_blocking=True)
        sun_direction = sun_direction.to(device=normals.device, dtype=normals.dtype, non_blocking=True)
        #sun_visibility = torch.where(sun_visibility < 0.95, torch.tensor(0.0), sun_visibility)
        intensity = direct_light_env_light.sun_intensity[None,...].repeat(incident_dirs.shape[0],1) #* 0.5
        #intensity = torch.ones_like(intensity) *torch.tensor([255, 178, 102]).to(device=intensity.device) * 3 / 255 #* 3
        sun_light = (sun_direction * normals).sum(dim=-1)[...,None] * intensity * 3#torch.ones_like(intensity) * 3 # * 10
        sun_light = sun_light.clip(0,10)
        f_d_ = (base_color / np.pi)
        f_s_ = GGX_specular(normals, viewdirs, sun_direction.unsqueeze(-2).detach(), roughness, fresnel=0.04).squeeze(-2)
        incident_sun_light = (f_d_ + f_s_) * sun_light * 0.3 
        pbr = pbr + incident_sun_light * sun_visibility 

    pbr = pbr 

    extra_results = {
        "diffuse_light":  diffuse_light, #specular
        "specular": specular,
        "incident_sun_light": sun_light ,
    }

    return pbr, extra_results

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

