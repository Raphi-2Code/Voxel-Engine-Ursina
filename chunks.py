from ursina import *
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
window.title="Minecraft Java Edition v1.21"
app=Ursina()
player=FirstPersonController(gravity=0,x=0,y=0)
window.borderless=False
cube_faces=[(0,1,0,180,0,0),(0,2,0,0,0,0),(0,1.5,0.5,90,0,0),(0,1.5,-0.5,-90,0,0),(0.5,1.5,0,0,0,90),(-0.5,1.5,0,0,0,-90)]
class Perlin:
    def __init__(self):
        self.seed = ord('y')+ord('o')
        self.octaves = 0.5
        self.freq = 8
        self.amplitude = 1

        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        return self.pNoise([x/self.freq, z/self.freq]) * self.amplitude
window.color=color.azure#rgb(51,3,3)echte nether-farbe
chunk_faces=[]
chunk_faces2=[]
Text("Minecraft Java Edition v1.21",y=.5,x=-.886,color=color.black)
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
noise=Perlin()
terrain=Entity()
for x in range(60):
    for z in range(60):
        y = noise.get_height(x,z)
        y = math.floor(y * 7.5)
        elem=cube_faces[1]
        pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
        rot_i = Vec3(elem[3], elem[4], elem[5])
        face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
        chunk_faces.append(face)
        chunk_faces2.append(face.position)
        if pos_i+(0,1,-1) in chunk_faces2:
            elem = cube_faces[2]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i+(0,1,-1), rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
        if pos_i+(-1,-1,0) in chunk_faces2:
            elem = cube_faces[5]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
        if pos_i+(0,-1,-1) in chunk_faces2:
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
        if pos_i+(-1,1,0) in chunk_faces2:
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i+(-1,1,0), rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
        if pos_i+(0.5,-0.5,-1) in chunk_faces2:
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
        if pos_i+(-1,1,0) in chunk_faces2:
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            if pos_i+(-1.5,1.5,-1) in chunk_faces2:
                face = Entity(model="plane", position=pos_i+(-1,1,-1), rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
terrain.combine()
terrain.texture="ursina-tutorials-main/assets/sandMinecraft.jfif"
app.run()
