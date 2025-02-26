from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

app = Ursina()

wall_thickness = 1
wall_height = 3
world_size = 10

ground = Entity(
    model='plane',
    scale=(2 * world_size, 1, 2 * world_size),
    texture='white_cube',
    texture_scale=(2 * world_size, 2 * world_size),
    collider='box'
)

left_wall = Entity(
    model='cube',
    scale=(wall_thickness, wall_height, 2 * world_size),
    position=(-world_size, wall_height / 2, 0),
    texture='white_cube',
    texture_scale=(2 * world_size, wall_height),
    collider='box'
)

right_wall = Entity(
    model='cube',
    scale=(wall_thickness, wall_height, 2 * world_size),
    position=(world_size, wall_height / 2, 0),
    texture='white_cube',
    texture_scale=(2 * world_size, wall_height),
    collider='box'
)

back_wall = Entity(
    model='cube',
    scale=(2 * world_size, wall_height, wall_thickness),
    position=(0, wall_height / 2, -world_size),
    texture='white_cube',
    texture_scale=(2 * world_size, wall_height),
    collider='box'
)

front_wall = Entity(
    model='cube',
    scale=(2 * world_size, wall_height, wall_thickness),
    position=(0, wall_height / 2, world_size),
    texture='white_cube',
    texture_scale=(2 * world_size, wall_height),
    collider='box'
)

sky = Sky()

player = FirstPersonController()

player.cursor.disable()

crosshair = Text(
    text='+',
    origin=(0, 0),
    scale=2,
    color=color.black50,
    position=(0, 0)
)
crosshair.parent = camera.ui

food = Entity(
    model="sphere",
    scale=(1,1,1),
    position=(3,1,3),
    color=color.red,
    collider="sphere"
)



def update():
    if held_keys['left mouse']:
        check_for_red()

def check_for_red():
    entity = mouse.hovered_entity
    if entity and hasattr(entity, 'color'):
        if entity.color == color.red:
            entity.blink(color.blue)





app.run()
