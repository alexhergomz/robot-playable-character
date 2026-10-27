# LISTEN-ONLY: records every service reply on the robot for a while.  Publishes nothing.
import time, sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Response_, Request_
ChannelFactoryInitialize(0, "eth0")
services = ["robot_state", "robot_type_service", "config", "motion_switcher", "sport", "arm",
            "videohub", "audiohub", "voice", "vui", "gpt", "slam_operate", "action_store",
            "basic_softlimit", "basic_taumax", "basic_demarcate", "dex3_msg_controller", "bashrunner"]
def mk(tag):
    def cb(m):
        try:
            api = m.header.identity.api_id
        except Exception:
            api = "?"
        data = getattr(m, "data", "")
        print(f"{time.strftime('%H:%M:%S')} {tag} api={api} {str(data)}", flush=True)
    return cb
subs = []
for s in services:
    for kind, typ in (("response", Response_), ("request", Request_)):
        sub = ChannelSubscriber(f"rt/api/{s}/{kind}", typ); sub.Init(mk(f"{s}/{kind}"), 10); subs.append(sub)
print(f"listening on {len(subs)} service topics for {sys.argv[1]} s", flush=True)
time.sleep(float(sys.argv[1]))
print("done", flush=True)
