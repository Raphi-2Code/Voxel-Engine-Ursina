# Voxel Engine Test

performance fix xd

The `cross-chunk-preview.py` demo now includes a more robust mining system that works across chunk borders, though performance may drop when mining

Codex, the AI coding agent by OpenAI helped me adding the advanced block placement system. I'm pretty impressed by this AI. Go to chatgpt.com/codex (requires a Teams subscription $25/mo, a Pro subscription $200/mo or an Enterprise subscription). You can also run Codex CLI locally via an API key.


![image](https://github.com/user-attachments/assets/db7e8cd7-e8c4-4677-9d40-0381433d305b)
This building system has a 12 block range (because Minecraft also has a 12 block range) and a cursor highlighting the block you can build on.


advanced building/mining system in development!

All the face checks are fixed!!! In the server and client!!!

![image](https://github.com/user-attachments/assets/fdd8eff8-1cee-4076-a285-2fa10bc93b98)

fixed!!!

The rotation of a face is a Vec3 with 6 params, this causes a bug, the faces are minimally "misrotated" by 2 or 1.5°, I have to fix this: Vec3(cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][3],cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][4],cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][5])

Imma put the updated file here when it's fixed

I wasn't working on my voxel engine for a long time, but today I was working on it again, so the update is here. - 04.03.2025; 19:09

![image](https://github.com/user-attachments/assets/9dcb5a2e-24a8-429a-b729-710fb52b7cd7)


A new pre-release (mining + building, but there are some chunk border bugs) - 13.02.2024; 20:30

![image](https://github.com/Raphi-2Code/Voxel-Engine-Ursina/assets/70066593/0deef8c0-bb4f-4e7e-8ca2-6bf6a40c05cb)


Now we have more terrain with updated chunk borders, after I added mining and building, I'm going to release it! - 04.02.2024; 19:16

1) install perlin_noise
2) run mining-update-working.py

# Known Bugs
```no bugs currently```

the mining update is out


the building update is out
![image](https://github.com/Raphi-2Code/chunk-checking-like-windsurftweeds-did/assets/70066593/d9dce256-788c-42cc-8306-0e4980eb006f)


New version
![image](https://github.com/Raphi-2Code/chunk-checking-like-windsurftweeds-did/assets/70066593/69f4444e-3b51-4c72-b48a-c129bc188d11)
//fixed checking for z axis



Old version
![image](https://github.com/Raphi-2Code/chunk-checking-like-windsurftweeds-did/assets/70066593/867c8bb0-b746-4d46-9322-033ea640cb9e)
//bad checking for z axis

