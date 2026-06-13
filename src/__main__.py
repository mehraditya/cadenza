"""Cadenza CLI — developer tools for Unitree robots.

Usage:
    cadenza list go1                                    # list available actions
    cadenza sim go1 "walk forward then jump"            # simulate in MuJoCo
    cadenza deploy go1 --ip 192.168.123.15              # deploy to real robot
"""

import click


@click.group()
def cli():
    """Cadenza — developer-first action library for Unitree robots."""
    pass


@cli.command("list")
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
def list_actions_cmd(robot: str):
    """List available actions for a robot."""
    from cadenza.actions import list_actions
    actions = list_actions(robot)
    click.echo(f"\n  {robot.upper()} — {len(actions)} actions\n")
    for name in sorted(actions):
        click.echo(f"    {name}")
    click.echo()


@cli.command()
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
@click.argument("command")
@click.option("--disturbance", "-d", type=float, default=None,
              help="Enable DisturbanceEngine with temperature 0.0–1.0 (e.g. -d 0.5)")
@click.option("--vla", is_flag=True, default=False,
              help="Enable VLA guardian for obstacle avoidance (uses SmolVLM-256M)")
@click.option("--obstacles", is_flag=True, default=False,
              help="Use obstacle course scene (boxes in the path)")
def sim(robot: str, command: str, disturbance: float | None, vla: bool, obstacles: bool):
    """Simulate actions in MuJoCo.

    \b
    Examples:
        cadenza sim go1 "walk forward then jump"
        cadenza sim g1 "stand then walk forward"
        cadenza sim go1 "walk forward" -d 0.5           # with disturbances
        cadenza sim go1 "walk forward 5 meters" --vla   # with VLA obstacle avoidance
        cadenza sim go1 "walk forward 5 meters" --vla --obstacles
    """
    if robot in ("go1", "go2"):
        xml_path = None
        if obstacles:
            from pathlib import Path
            xml_path = str(Path(__file__).resolve().parent.parent / "models" / "go1" / "obstacle_scene.xml")

        if vla:
            from cadenza.go1 import Go1
            go1 = Go1(xml_path=xml_path)
            from cadenza.parser import CommandParser
            parser = CommandParser(robot)
            calls = parser.parse(command)
            steps = [go1._call_to_step(c) for c in calls]
            go1.run(steps, vla=True)
        else:
            from cadenza.sim import Sim
            s = Sim(robot, xml_path=xml_path, disturbance=disturbance)
            if disturbance is not None:
                click.echo(f"  DisturbanceEngine ON (temperature={disturbance})")
            s.run(command)
    elif robot == "g1":
        from cadenza.g1 import G1
        from cadenza.parser import CommandParser
        g1 = G1()
        parser = CommandParser(robot)
        calls = parser.parse(command)
        steps = [g1._call_to_step(c) for c in calls]
        g1.run(steps)


@cli.group("action")
def action_group():
    """Inspect the action library (local actions + read-only built-ins)."""
    pass


@action_group.command("list")
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
@click.option("--project", type=click.Path(), default=None,
              help="Project directory containing local actions/*.json")
def action_list(robot: str, project: str | None):
    """List local actions and the read-only built-ins for a robot."""
    from pathlib import Path
    from cadenza.actions import action_builder as ab

    local = []
    if project:
        adir = Path(project) / "actions"
        if adir.is_dir():
            local = sorted(p.stem for p in adir.glob("*.json"))

    builtins = ab.list_builtin_actions(robot)

    click.echo(f"\n  {robot.upper()} actions\n")
    if project:
        click.echo("  local:")
        if local:
            for name in local:
                click.echo(f"    {name}")
        else:
            click.echo("    (none)")
        click.echo()
    if builtins:
        click.echo(f"  built-in (read-only): {', '.join(builtins)}")
    else:
        click.echo("  built-in (read-only): (none bundled)")
    click.echo()


@action_group.command("show")
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
@click.argument("name")
@click.option("--project", type=click.Path(), default=None,
              help="Project directory containing local actions/*.json")
def action_show(robot: str, name: str, project: str | None):
    """Show one action, resolving local first then the built-in library."""
    import json
    from cadenza.actions import action_builder as ab

    action, source = ab.resolve_action(project, robot, name)
    if action is None:
        click.echo(f"Action '{name}' not found for {robot}.", err=True)
        raise SystemExit(1)

    tag = " [read-only built-in]" if source == "builtin" else ""
    click.echo(f"\n  {name} ({robot}){tag}\n")
    click.echo(json.dumps(ab.build_action_payload(action), indent=2))
    click.echo()


@action_group.command("remove")
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
@click.argument("name")
@click.option("--project", type=click.Path(), default=None,
              help="Project directory containing local actions/*.json")
def action_remove(robot: str, name: str, project: str | None):
    """Remove a local action. Refuses on a read-only built-in."""
    from pathlib import Path
    from cadenza.actions import action_builder as ab

    if ab.is_builtin_action(robot, name):
        click.echo(f"{name} is a read-only built-in; it can't be edited or "
                   f"removed.", err=True)
        raise SystemExit(1)

    if not project:
        click.echo("Error: --project required to remove a local action", err=True)
        raise SystemExit(1)

    local = Path(project) / "actions" / f"{name}.json"
    if not local.is_file():
        click.echo(f"Local action '{name}' not found in {project}.", err=True)
        raise SystemExit(1)
    local.unlink()
    click.echo(f"Removed local action '{name}'.")


@cli.command()
@click.argument("robot", type=click.Choice(["go1", "go2", "g1"]))
@click.option("--ip", required=True, help="Robot IP address")
@click.option("--key", default="~/.ssh/id_rsa", help="SSH key path")
@click.option("--mode", type=click.Choice(["ssh", "direct", "bridge"]),
              default="ssh", help="Deployment mode")
@click.option("--command", "-c", default=None,
              help="Action command string (e.g. 'walk forward then jump')")
def deploy(robot: str, ip: str, key: str, mode: str, command: str | None):
    """Deploy actions to a real robot.

    \b
    Examples:
        cadenza deploy go1 --ip 192.168.123.15 -c "walk forward then sit"
        cadenza deploy g1 --ip 10.0.0.1 --mode direct -c "stand then walk forward"
    """
    if robot in ("go1", "go2"):
        from cadenza.go1 import Go1
        go1 = Go1()
        if mode == "ssh":
            click.echo(f"Deploying to {robot.upper()} at {ip} via SSH...")
            go1.deploy_ssh(None, host=ip, key=key, command=command)
        elif mode == "direct":
            if not command:
                click.echo("Error: --command/-c required for direct mode", err=True)
                raise SystemExit(1)
            from cadenza.parser import CommandParser
            parser = CommandParser(robot)
            calls = parser.parse(command)
            steps = [go1._call_to_step(c) for c in calls]
            go1.deploy(steps)
        elif mode == "bridge":
            bridge = go1.deploy_ssh_bridge(host=ip, key=key)
            click.echo(f"Bridge connected to {ip}. Use bridge.send_action().")
    elif robot == "g1":
        from cadenza.g1 import G1
        g1 = G1()
        if mode == "ssh":
            click.echo(f"Deploying to G1 at {ip} via SSH...")
            g1.deploy_ssh(None, host=ip, key=key, command=command)
        elif mode == "direct":
            if not command:
                click.echo("Error: --command/-c required for direct mode", err=True)
                raise SystemExit(1)
            from cadenza.parser import CommandParser
            parser = CommandParser(robot)
            calls = parser.parse(command)
            steps = [g1._call_to_step(c) for c in calls]
            g1.deploy(steps)
        elif mode == "bridge":
            bridge = g1.deploy_ssh_bridge(host=ip, key=key)
            click.echo(f"Bridge connected to {ip}. Use bridge.send_action().")


if __name__ == "__main__":
    cli()
