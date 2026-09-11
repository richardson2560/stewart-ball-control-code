# tools/inspect_coppelia_scene.py

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def inspect_scene():
    client = RemoteAPIClient(host="localhost", port=23000)
    sim = client.require('sim')
    
    scene_name = sim.getStringParam(sim.stringparam_scene_name)
    print(f"[+] Connected to CoppeliaSim.")
    print(f"[+] Current Scene: '{scene_name}'")
    
    # Check key handles with error suppression
    objects_to_test = [
        "/stewartPlatform", "stewartPlatform",
        "/stewartPlatform/tip", "stewartPlatform/tip",
        "/Sphere", "Sphere",
        "/platformTable", "platformTable",
        "/stewartPlatform/motor1", "stewartPlatform/motor1"
    ]
    
    print("\n" + "=" * 50)
    print(f"{'Path / Alias':<30} | {'Status':<15}")
    print("=" * 50)
    for path in objects_to_test:
        handle = sim.getObject(path, {'noError': True})
        status = f"FOUND (id={handle})" if handle != -1 else "NOT FOUND"
        print(f"{path:<30} | {status:<15}")
    print("=" * 50)

if __name__ == "__main__":
    inspect_scene()