# sandbox/executor.py
import subprocess
from pathlib import Path


class SandboxTimeout(Exception):
    pass


def start_sandbox(
    image: str,
    workdir: Path,
    cmd: list[str],
    name: str,
    network: str,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
) -> subprocess.Popen:
    """
    Launch `cmd` inside a disposable container and return the Popen
    handle for the `docker run` wrapper process immediately (non-blocking).

    Caller is responsible for communicate()/timeout and for killing by
    `name` via kill_sandbox() -- killing the returned Popen only kills
    the `docker run` CLI, NOT the container itself.

    `network` is required, not defaulted: omitting --network entirely
    puts a container on Docker's default bridge, which HAS outbound
    internet access -- the opposite of what a "no network" caller would
    expect. Callers must say "none" or an explicit network name.

    `env` is an explicit allowlist, never a copy of the caller's
    environment. subprocess.Popen does not forward the host env into
    the container on its own (docker run only sees --env you pass), so
    the only way a secret crosses into the sandbox is a future caller
    passing it here or via --env-file -- which is exactly the failure
    mode test_no_secrets_in_sandbox_env exists to catch.

    `volumes` is {mount_source: container_path} for anything beyond the
    always-present /repo mount -- a named docker volume (the install
    cache) or a host directory (patch extraction's /out). Caller decides
    which; this function doesn't care which kind of string it's handed.
    """
    docker_cmd = [
        "docker", "run",
        "--rm",
        "--name", name,
        "--memory=2g",
        "--cpus=2",
        "--pids-limit=512",
        "--network", network,
        "-v", f"{workdir}:/repo",
        "-w", "/repo",
    ]
    for source, container_path in (volumes or {}).items():
        docker_cmd += ["-v", f"{source}:{container_path}"]
    for key, value in (env or {}).items():
        docker_cmd += ["--env", f"{key}={value}"]
    docker_cmd += [image, *cmd]

    return subprocess.Popen(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def kill_sandbox(name: str) -> None:
    """Kill a container by its deterministic name. Safe to call even
    if the container has already exited -- docker kill on a gone
    container just errors, which we swallow."""
    subprocess.run(["docker", "kill", name], capture_output=True)