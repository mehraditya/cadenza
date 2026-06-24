<h1 align="center" style="margin-bottom: 8px;">
  <img alt="Cadenza" src="https://raw.githubusercontent.com/aparekh02/cadenza/main/cadenza-logo.png" height="70">
</h1>

<h3 align="center" style="margin-top: 16px; margin-bottom: 16px;">
Run and deploy complex robot actions with a simple Python SDK.
</h3>

<p align="center">
  <a href="#-supported-robots">Robots</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="https://www.cadenzalabs.xyz">Docs</a> •
  <a href="DEPLOY.md">Deploy</a>
</p>

Cadenza lets you write complex motion-targeted code and run it in MuJoCo or on
hardware. Three robots are supported out of the box.

**Website:** [www.cadenzalabs.xyz](http://www.cadenzalabs.xyz) &nbsp;•&nbsp; **pip package:** [cadenza-lab](https://pypi.org/project/cadenza-lab/) &nbsp;•&nbsp; **Deploy to hardware:** [DEPLOY.md](DEPLOY.md)

## 🤖 Supported robots

| Robot | Type | Actions | Control model |
|-------|------|---------|---------------|
| **Go1** | Unitree quadruped | 21 | Gaits + phases (`walk`, `trot`, `turn`, `jump`, `sit`, …) |
| **G1** | Unitree humanoid | 20 | Bipedal locomotion + arms |
| **Arm** | 6-axis articulated arm | 6 | Cartesian `move_to` / `pick` / `place` (IK + grasp) |

> Go1 and G1 deploy to real hardware (see [DEPLOY.md](DEPLOY.md)); the arm is
> simulation-only today. Want **drones, other arms, or custom hardware**? Reach
> out — [acparekh@stanford.edu](mailto:acparekh@stanford.edu).

## ⚡ Quickstart

```bash
pip install cadenza-lab
```

Each robot follows the same shape: create a controller, build a list of actions,
`run()`. The clips below are each rendered straight from the snippet above them.

### 🐾 Go1 — quadruped

The Go1 stands, walks 2 m, arcs through a turn (nested list = concurrent),
jumps, and sits.

```python
import cadenza_lab as cadenza

go1 = cadenza.go1()
go1.run([
    go1.stand(),
    go1.walk_forward(speed=1.5, distance_m=2.0),
    [go1.turn_left(), go1.walk_forward()],   # concurrent: walking arc
    go1.jump(speed=2.0, extension=1.2),
    go1.sit(),
])
```

<p align="center">
  <img alt="Go1 quickstart" src="https://raw.githubusercontent.com/aparekh02/cadenza/main/assets/go1.gif" width="420">
</p>

### 🦾 G1 — humanoid

The G1 stands and jumps under active balance stabilization, landing back on
its feet each time.

```python
import cadenza_lab as cadenza

g1 = cadenza.g1()
g1.run([
    g1.stand(),
    g1.jump(),
    g1.stand(),
    g1.jump(),
    g1.stand(),
])
```

<p align="center">
  <img alt="G1 quickstart" src="https://raw.githubusercontent.com/aparekh02/cadenza/main/assets/g1.gif" width="420">
</p>

### 🤖 Arm — 6-axis pick & place

The arm homes, picks the cube off the table, places it to the side, and returns
home. Targets are Cartesian `(x, y, z)`; motion is IK-driven and the grasp is a
weld the controller activates when the gripper closes.

```python
import cadenza_lab as cadenza

arm = cadenza.arm()
arm.run([
    arm.home(),
    arm.pick((0.50, 0.00, 0.43)),   # grab the cube on the table
    arm.place((0.40, 0.22, 0.43)),  # set it down to the side
    arm.home(),
])
```

<p align="center">
  <img alt="Arm quickstart" src="https://raw.githubusercontent.com/aparekh02/cadenza/main/assets/arm.gif" width="420">
</p>

## 📚 Next steps

| | |
|---|---|
| **Full SDK docs** | [www.cadenzalabs.xyz](https://www.cadenzalabs.xyz) — robots, actions, scenes, gym adapter, inference stack, multi-robot coordination |
| **Deploy to hardware** | [DEPLOY.md](DEPLOY.md) — SSH, DDS direct, and bridge modes for Go1 / G1 |
| **CLI** | `cadenza list go1` · `cadenza sim go1 "walk forward then jump"` · `cadenza list arm` |
| **Examples** | [`example.py`](example.py), [`examples/`](examples/) |

## 💚 Community

| | Links |
|---|---|
| **GitHub** | [aparekh02/cadenza](https://github.com/aparekh02/cadenza) |
| **Issues** | [Report a bug or request a feature](https://github.com/aparekh02/cadenza/issues) |
| **License** | [Apache 2.0](LICENSE) |
