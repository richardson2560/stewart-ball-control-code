# src/sbc/interfaces/headless_launcher.py

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional


class CoppeliaLauncher:
    """
    Manages detection, background execution, and lifecycle of the CoppeliaSim process.
    """

    @staticmethod
    def find_executable() -> str:
        """
        Locates the CoppeliaSim binary across Windows, Linux, and macOS default paths.
        Override using the COPPELIASIM_ROOT environment variable if installed in a custom path.
        """
        env_root = os.environ.get("COPPELIASIM_ROOT")
        sys_type = platform.system()

        if sys_type == "Windows":
            if env_root and (Path(env_root) / "coppeliaSim.exe").exists():
                return str(Path(env_root) / "coppeliaSim.exe")
            
            candidates = [
                Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"),
                Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimPRO\coppeliaSim.exe"),
                Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimPlayer\coppeliaSim.exe"),
            ]
            for p in candidates:
                if p.exists():
                    return str(p)

        elif sys_type == "Linux":
            if env_root and (Path(env_root) / "coppeliaSim.sh").exists():
                return str(Path(env_root) / "coppeliaSim.sh")
            
            candidates = [
                Path("/usr/local/bin/coppeliasim"),
                Path("/usr/local/CoppeliaSim/coppeliaSim.sh"),
                Path.home() / "CoppeliaSim" / "coppeliaSim.sh",
            ]
            for p in candidates:
                if p.exists():
                    return str(p)

        elif sys_type == "Darwin":
            if env_root and (Path(env_root) / "coppeliaSim").exists():
                return str(Path(env_root) / "coppeliaSim")
            
            app_path = Path("/Applications/CoppeliaSim.app/Contents/MacOS/coppeliaSim")
            if app_path.exists():
                return str(app_path)

        fallback = "coppeliaSim.exe" if sys_type == "Windows" else "coppeliaSim.sh"
        return fallback

    @staticmethod
    def resolve_scene_path(scene_name: str = "stewart_platform.ttt") -> Path:
        """
        Resolves the absolute path to the CoppeliaSim scene file within simulations/coppelia/.
        """
        current = Path(__file__).resolve()
        
        # Traverse upward until the project root is located
        project_root = None
        for parent in [current] + list(current.parents):
            if (parent / "simulations").is_dir() and (parent / "src").is_dir():
                project_root = parent
                break

        if project_root is None:
            project_root = current.parents[2]

        candidate = project_root / "simulations" / "coppelia" / scene_name
        if candidate.exists():
            return candidate.resolve()

        # Fallback shallow search
        for path in project_root.rglob(scene_name):
            if path.is_file():
                return path.resolve()

        raise FileNotFoundError(
            f"Could not locate CoppeliaSim scene '{scene_name}' under {project_root}."
        )

    @classmethod
    def start(
        cls,
        scene_name: str = "stewart_platform.ttt",
        headless: bool = True,
        port: int = 23000
    ) -> subprocess.Popen:
        """
        Spawns the CoppeliaSim subprocess with remote ZeroMQ enabled.
        """
        executable = cls.find_executable()
        scene_path = cls.resolve_scene_path(scene_name)

        cmd = [executable]
        if headless:
            cmd.append("-h")
        cmd.append(str(scene_path))

        env = os.environ.copy()
        if port != 23000:
            env["ZMQ_REMOTE_API_PORT"] = str(port)

        process = subprocess.Popen(cmd, cwd=str(scene_path.parent), env=env)
        return process

    @staticmethod
    def stop(process: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
        """Terminates the CoppeliaSim subprocess safely."""
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()