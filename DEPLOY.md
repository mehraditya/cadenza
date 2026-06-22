# Deploying Cadenza to a real robot

The same `Step` sequences you simulate run on physical hardware. Cadenza targets
the Unitree **Go1** (quadruped) and **G1** (humanoid) today. This guide covers
the three deployment modes and the CLI/SDK entry points for each.

> Want support for **drones, robot arms, or other hardware**? The 6-axis arm is
> simulation-only for now. Reach out — [acparekh@stanford.edu](mailto:acparekh@stanford.edu).

## Modes at a glance

| Mode | What it does | When to use it |
|------|--------------|----------------|
| **SSH** | Upload and run a script on the robot's onboard computer | Self-contained routines that run entirely on the robot |
| **DDS direct** | Send motor commands from your laptop over DDS (same network) | Driving the robot live from a laptop on the robot's network |
| **Bridge** | Heavy compute on your laptop, lightweight actions on the robot | Model-in-the-loop control (VLA on your GPU, actions on the robot) |

## CLI

```bash
cadenza deploy go1 --ip 192.168.123.15 -c "walk forward then sit"   # SSH (default)
cadenza deploy go1 --ip 192.168.123.15 --mode direct -c "..."       # DDS direct
cadenza deploy go1 --ip 192.168.123.15 --mode bridge                # bridge mode
```

## SSH deploy

Upload a script and run it on the robot's onboard computer — everything executes
on the robot.

```python
import cadenza

go1 = cadenza.go1()
go1.deploy_ssh(
    "examples/unitree_go1/deploy_go1.py",
    host="192.168.123.15",
    key="~/.ssh/go1_rsa",
)
```

## DDS direct

Send motor commands directly over DDS from a laptop on the robot's network — no
SSH round-trip.

```python
go1 = cadenza.go1()
go1.deploy([
    go1.stand(),
    go1.walk_forward(speed=0.5),
    go1.sit(),
])
```

## Bridge mode

Bridge mode keeps a live control handle so your laptop can run the heavy model
while the robot runs the lightweight action engine.

```python
go1 = cadenza.go1()
bridge = go1.deploy_ssh_bridge(host="192.168.123.15", key="~/.ssh/go1_rsa")

# Model-in-the-loop: your laptop GPU runs inference, the robot runs actions.
while True:
    state = bridge.telemetry
    if state and state.joint_q:
        action = my_model(state)
        bridge.send_action(action, speed=0.5)

bridge.estop()
```

## Per-robot examples

| Robot | Example |
|-------|---------|
| Go1 | [`examples/unitree_go1/deploy_go1.py`](examples/unitree_go1/deploy_go1.py) — SSH / DDS / bridge |
| G1 | [`examples/unitree_g1/deploy_g1.py`](examples/unitree_g1/deploy_g1.py) — sim and deployment |

## Safety

- Always have the robot on a stand or in a clear, padded area for first runs.
- `bridge.estop()` cuts motor commands immediately — keep it reachable in your
  control loop.
- Test every sequence in simulation (`cadenza sim ...` or `robot.run([...])`)
  before deploying it to hardware.

---

← Back to the [README](README.md).
