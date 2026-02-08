# Tracking Cones

![alt text](docs/110-cone-mask.gif)

A possibly misguided attempt at a camera positioning system using 3D printing, Blender, and Python.

![alt text](docs/220-cone-contours.gif)

## Setup

Install Docker

```bash
docker compose up -d
```

Exec into tracking_server container and run the renderer:
```bash
docker exec -it tracking_server bash
blender --background --script renderer.py
```

Convert the renders to a GIF:
```bash
docker exec -it tracking_server bash
python renders-to-gif.py
```

## Idea sketches

![alt text](docs/tracking-cone-diagram.png)
![alt text](docs/tracking-cone-diagram-localization.png)
![alt text](docs/relative-angle-calculation.png)
