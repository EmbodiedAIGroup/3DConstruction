import time
import math
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from queue import Queue

class Go2Controller:
    def __init__(self, ifname="enx607d099f16d2", topic="rt/lf/sportmodestate"):
        # 状态缓存队列
        self.latest_state = Queue(maxsize=1)
        # 动作控制变量
        self.current_action = 0  # 0-停止,1-前进,2-左转,3-右转
        self.current_velocity = 0.1
        self.current_angle = math.pi / 12
        self.current_duration = 1.0
        self.last_action_time = time.time()
        self.action_lock = threading.Lock()
        self.keep_running = False

        # 初始化机器人连接
        self.sport_client, self.state_sub = self._init_agent(ifname, topic)
        
        # 启动动作发送线程
        self.action_thread = threading.Thread(target=self._keep_sending_action)
        self.action_thread.daemon = True
        self.action_thread.start()

    def _init_agent(self, ifname, topic):
        """初始化机器人连接和状态订阅"""
        ChannelFactoryInitialize(0, ifname)
        sport_client = SportClient()
        sport_client.SetTimeout(10.0)
        sport_client.Init()

        # 状态回调函数
        def msg_handler(msg: SportModeState_):
            if self.latest_state.full():
                self.latest_state.get_nowait()
            self.latest_state.put(msg)

        # 订阅状态话题
        state_sub = ChannelSubscriber(topic, SportModeState_)
        state_sub.Init(msg_handler, 10)
        return sport_client, state_sub

    def _send_action(self, action):
        """发送单个动作指令"""
        if action == 1:
            self.sport_client.Move(vx=self.current_velocity, vy=0, vyaw=0)
        elif action == 2:
            self.sport_client.Move(vx=self.current_velocity, vy=0, vyaw=self.current_angle)
        elif action == 3:
            self.sport_client.Move(vx=self.current_velocity, vy=0, vyaw=-self.current_angle)
        else:  # 0-停止
            self.sport_client.Move(vx=0, vy=0, vyaw=0)

    def _keep_sending_action(self):
        """持续发送动作的后台线程"""
        self.keep_running = True
        while self.keep_running:
            with self.action_lock:
                # 检查动作是否超时
                time_since_last = time.time() - self.last_action_time
                if time_since_last > self.current_duration:
                    self.current_action = 0  # 超时自动停止
                # 发送当前动作
                self._send_action(self.current_action)
            time.sleep(0.05)  # 20Hz发送频率

    def set_action(self, action, velocity=0.1, angle=math.pi/12, duration=1.0):
        """设置动作参数（线程安全）"""
        with self.action_lock:
            self.current_action = action
            self.current_velocity = velocity
            self.current_angle = angle
            self.current_duration = duration
            self.last_action_time = time.time()

    def get_pose(self):
        """获取机器人当前位姿（位置+姿态）"""
        if not self.latest_state.empty():
            state = self.latest_state.get()
            return {
                "position": [state.position[0], state.position[1], state.position[2]],
                "orientation": [state.imu_state.rpy[0], state.imu_state.rpy[1], state.imu_state.rpy[2]],
                "timestamp": ''
            }
        return None

    def stop(self):
        """紧急停止"""
        self.set_action(0)

    def disconnect(self):
        """断开连接"""
        self.keep_running = False
        self.action_thread.join()
        self.stop()