from ursina import *
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
window.title="Minecraft Java Edition v1.20.1"
app=Ursina()
player=FirstPersonController(gravity=0)
window.borderless=False
#EditorCamera()
cube_faces=[(0,1,0,180,0,0),(0,2,0,0,0,0),(0,1.5,0.5,90,0,0),(0,1.5,-0.5,-90,0,0),(0.5,1.5,0,0,0,90),(-0.5,1.5,0,0,0,-90)]
noise = PerlinNoise(octaves=3, seed=1234)
window.color=color.azure#rgb(51,3,3) nether color
chunk_faces={}
terrain=Entity()
Text("Minecraft Java Edition v1.20.1",y=.5,x=-.886,color=color.black)
window.fps_counter.disable()
window.cog_menu.disable()
window.cog_button.disable()
window.exit_button.disable()
def input(key):
    if key=="m": player.y+=1
    if key=="y": player.y-=1
    if key=="u": print(player.position)
Entity(model="cube",x=1,color=color.red)
Entity(model="cube",z=1,color=color.yellow)
for z in range(60):
    for x in range(60):
        y = noise([x * 0.02, z * 0.02])
        y = math.floor(y * 7.5)
        elem=cube_faces[1]
        pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
        rot_i = Vec3(elem[3], elem[4], elem[5])
        face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)#texture="white_cube",
        chunk_faces[face] = pos_i
        if pos_i+(0,1,-1) in chunk_faces.values():
            elem = cube_faces[2]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i+(0,1,-1), rotation=rot_i, parent=terrain)  # texture="white_cube",
            chunk_faces[face] = pos_i
        if pos_i+(-1,-1,0) in chunk_faces.values():
            elem = cube_faces[5]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)  # texture="white_cube",
            chunk_faces[face] = pos_i
        if pos_i+(0,-1,-1) in chunk_faces.values():
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)  # texture="white_cube",
            chunk_faces[face] = pos_i
        if pos_i+(-1,1,0) in chunk_faces.values():
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i+(-1,1,0), rotation=rot_i, parent=terrain)  # texture="white_cube",
            chunk_faces[face] = pos_i
            if pos_i+(1,0,-2) in chunk_faces.values() or pos_i+(1,0,-1) in chunk_faces.values() or pos_i+(2,1,-2) in chunk_faces.values() or pos_i+(2,1,-1) in chunk_faces.values():
                elem = cube_faces[4]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face2 = Entity(model="plane", position=pos_i + (-1, 1, -1), rotation=rot_i,
                              parent=terrain)  # texture="white_cube",

        if pos_i+(0.5,-0.5,-1) in chunk_faces.values():
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i,
                          parent=terrain)

        if pos_i+(-1,1,-1) in chunk_faces.values():
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i+(-1,1,-1), rotation=rot_i,
                      parent=terrain,color=color.white)  # texture="white_cube",
            chunk_faces[face] = pos_i
terrain.combine()
terrain.texture="sandMinecraft.jfif"
#terrain.collider='mesh'
#EditorCamera()
app.run()
