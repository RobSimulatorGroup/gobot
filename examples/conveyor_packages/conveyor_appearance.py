"""Deterministic parcel artwork and ordinary scene material resources.

The label is part of the film's UV texture, so it follows the simulated surface.
All codes are invented demonstration markings; there is no recipient data.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _artwork(kind, *, label=True):
    size = 256
    color = {'silver': (242, 244, 242), 'yellow': (244, 197, 20), 'carton': (168, 126, 76)}[kind]
    rng = np.random.default_rng(71)
    grain = rng.normal(0, .3 if kind != 'carton' else 3.5, (size, size, 1))
    image = Image.fromarray(np.clip(np.array(color)[None, None] + grain, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=9)
    if kind == 'carton':
        draw.text((37, 102), 'EXPRESS', fill=(39, 34, 27), font=ImageFont.load_default(size=32))
        draw.text((39, 144), 'HANDLE WITH CARE', fill=(52, 44, 32), font=small)
        for x in (76, 102):
            draw.line((x, 87, x, 48), fill=(52, 44, 32), width=3)
            draw.polygon(((x-7, 56), (x, 44), (x+7, 56)), fill=(52, 44, 32))
    elif label:
        # Fine parallel heat-seal lines around the pillow flange.
        for inset in (2, 4):
            draw.rectangle((inset, inset, size-inset-1, size-inset-1), outline=tuple(int(c*.94) for c in color), width=1)
        draw.rounded_rectangle((57, 68, 201, 176), radius=2, fill=(246, 246, 242), outline=(212, 213, 207))
        draw.text((66, 74), 'STANDARD PARCEL', fill=(76, 79, 80), font=small)
        x = 68
        while x < 189:
            width = int(rng.integers(1, 3))
            draw.rectangle((x, 91, x+width, 117), fill=(32, 36, 39))
            x += width + int(rng.integers(1, 3))
        draw.text((66, 123), 'GB 2048 0071 03', fill=(62, 65, 66), font=small)
        draw.text((66, 140), 'EXPRESS\nDEMO PARCEL', fill=(82, 85, 86), font=small)
        draw.text((173, 132), 'A1' if kind == 'silver' else 'A3', fill=(40, 43, 44), font=font)
        for iy in range(7):
            for ix in range(7):
                if rng.integers(0, 2):
                    draw.rectangle((174+ix*2, 151+iy*2, 175+ix*2, 152+iy*2), fill=(68, 72, 73))
    # Mesh v=0 is at the bottom; stored image rows begin at the top.
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _film_maps(kind):
    """Two UV panels: a labelled face and a plain reverse, with soft creases.

    Wrinkles change the surface normal instead of painting fixed highlights
    into the plastic. The paper label has a smoother normal and matte finish.
    """
    size = 512
    v, u = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    normal_panels, roughness_panels = [], []
    physical_size = (.42, .30) if kind == 'silver' else (.29, .205)
    for panel in range(2):
        rng = np.random.default_rng((71 if kind == 'silver' else 113) + panel)
        height = np.zeros_like(u)
        # Short, curved folds radiate inward from the welded perimeter.
        for along, inward in ((u, v), (u, 1-v), (v, u), (v, 1-u)):
            for start in np.linspace(.06, .94, 12) + rng.uniform(-.018, .018, 12):
                length = rng.uniform(.12, .29)
                curve = start + rng.uniform(-.24, .24) * inward + .012 * np.sin(16 * inward + rng.uniform(0, 6))
                width = rng.uniform(.006, .012)
                taper = np.sin(np.pi * np.clip(inward / length, 0., 1.)) ** 1.3
                height += rng.uniform(.00035, .00075) * np.exp(-((along-curve)/width)**2) * taper
        # Irregular slack folds span the face. Their varying lengths and curved
        # paths avoid the regular ribs of an inflated cushion. The physical
        # shell still controls silhouette and the larger contact deformations.
        for offset, slope, bend, amplitude in (
                (.18, .30, .24, .0011), (.29, .22, -.16, .0017),
                (.41, -.16, .17, .0010), (.58, -.23, -.14, .0014),
                (.70, .10, -.22, .0018), (.83, -.14, .18, .0010)):
            offset += rng.uniform(-.025, .025)
            curve = offset + slope*(u-.5) + bend*(u-.5)**2 + .016*np.sin(5*u + offset*13)
            width = rng.uniform(.014, .022)
            along = np.sin(np.pi*u)**rng.uniform(.5, 1.4)
            height += amplitude * np.exp(-((v-curve)/width)**2) * along
            height -= .35*amplitude * np.exp(-((v-curve-1.4*width)/width)**2) * along
        edge = np.minimum.reduce((u, 1-u, v, 1-v))
        height *= np.clip(edge/.015, 0., 1.)
        # The postage label follows the simulated bag but suppresses fine folds.
        paper = ((u > 57/256) & (u < 202/256) & (v > 79/256) & (v < 188/256)) if panel == 0 else np.zeros_like(u, dtype=bool)
        dy, dx = np.gradient(height, physical_size[1]/(size-1), physical_size[0]/(size-1))
        dx = np.where(paper, .35*dx, dx)
        dy = np.where(paper, .35*dy, dy)
        normals = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        normal_panels.append(np.rint((normals*.5+.5)*255).astype(np.uint8))
        roughness = .49 + .025*np.sin(9*u+6*v) + .018*np.cos(14*v-4*u)
        roughness += rng.normal(0., .003, u.shape)
        roughness = np.where(paper, .78, np.clip(roughness, .43, .55))
        packed = np.zeros((*u.shape, 3), dtype=np.uint8)
        packed[..., 0] = 255
        packed[..., 1] = np.rint(roughness*255).astype(np.uint8)
        roughness_panels.append(packed)
    albedo = Image.new('RGB', (2*size, size))
    albedo.paste(_artwork(kind).resize((size, size), Image.Resampling.LANCZOS), (0, 0))
    albedo.paste(_artwork(kind, label=False).resize((size, size), Image.Resampling.LANCZOS), (size, 0))
    return {'albedo': albedo,
            'normal': Image.fromarray(np.concatenate(normal_panels, axis=1)),
            'roughness': Image.fromarray(np.concatenate(roughness_panels, axis=1))}


def _store_texture(scene, assets, name, image):
    # Native Image resources keep authoring independent of the editor importer.
    resource = {'__VERSION__': 3, '__META_TYPE__': 'RESOURCE', '__TYPE__': 'Image',
                '__SUB_RESOURCES__': [], '__EXT_RESOURCES__': [],
                '__RESOURCE__': {'storage': {'width': image.width, 'height': image.height,
                'use_mipmaps': False, 'format': 'RGB8', 'data': list(image.tobytes())}}}
    destination = assets / f'{name}.jres'
    text = json.dumps(resource, separators=(',', ':')) + '\n'
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
    scene['__EXT_RESOURCES__'].append({'__ID__': f'parcel_image_{name}', '__TYPE__': 'Image',
                                      '__PATH__': f'res://assets/parcels/{name}.jres'})
    scene['__SUB_RESOURCES__'].append({'__ID__': f'parcel_texture_{name}', '__TYPE__': 'Texture2D',
                'image': f'ExtResource(parcel_image_{name})', 'min_filter': 'Linear', 'mag_filter': 'Linear',
                'wrap_u': 'ClampToEdge', 'wrap_v': 'ClampToEdge'})
    return f'SubResource(parcel_texture_{name})'


def apply_appearance(scene, assets):
    assets = Path(assets)
    assets.mkdir(parents=True, exist_ok=True)
    sub = scene['__SUB_RESOURCES__']
    for kind in ('silver', 'yellow', 'carton'):
        maps = {'albedo': _artwork(kind)} if kind == 'carton' else _film_maps(kind)
        textures = {channel: _store_texture(scene, assets, kind if channel == 'albedo' else f'{kind}_{channel}', image)
                    for channel, image in maps.items()}
        detail = {} if kind == 'carton' else {
            'normal_texture': textures['normal'], 'normal_scale': 1.,
            'metallic_roughness_texture': textures['roughness'], 'specular': .36}
        sub.append({'__ID__': f'parcel_material_{kind}', '__TYPE__': 'PBRMaterial3D',
                    'albedo': {'red': 1., 'green': 1., 'blue': 1., 'alpha': 1.},
                    'albedo_texture': textures['albedo'],
                    'roughness': .86 if kind == 'carton' else 1., **detail,
                    'metallic': 0., 'double_sided': kind != 'carton', 'alpha_mode': 'Opaque'})
    for node in scene['__NODES__']:
        name = node['name']
        if name in ('soft_mailer_blue', 'soft_pouch_yellow'):
            kind = 'silver' if name == 'soft_mailer_blue' else 'yellow'
            node['properties']['material'] = f'SubResource(parcel_material_{kind})'
        elif name in ('carton_small_visual', 'carton_wide_visual', 'carton_tall_visual'):
            node['properties']['material'] = 'SubResource(parcel_material_carton)'
