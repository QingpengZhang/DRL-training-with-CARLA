## human demonstration with joystick for agent training
# store the human demonstration in the buffer
import sys
import os

sys.path.append('D:/02_Envs/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.15-py3.7-win-amd64.egg')
sys.path.append('D:/02_Envs/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/')

import carla
import numpy as np
import numpy as np
import random
import pygame
import cv2
import time
import threading
image_lock = threading.Lock()

try:
    # from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.behavior_agent import BehaviorAgent
except ModuleNotFoundError:
    print("Error: 'agents.navigation.basic_agent' module not found. Ensure it is installed and accessible.")
    sys.exit(1)


# --- Data Storage ---
DATA_PATH_HUMAN = "human_driving_data"
DATA_PATH_AGENT = "agent_driving_data"
IMAGES_PATH = os.path.join(DATA_PATH_AGENT, "images")
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
actions_log = open(os.path.join(DATA_PATH_AGENT, "actions.csv"), "w")
actions_log.write("timestamp,image_file,throttle,steer,brake\n")
frame_id = 0
FULLBLACK = (255, 255, 255)
FULLGREEN = (0, 255, 0)
FULLRED = (255, 0, 0)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town05')
map = world.get_map()
original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # Fixed time step for simulation
world.apply_settings(settings)


def run_collection_data(world, 
                        agent, 
                        vehicle,  
                        episode_id, 
                        IMAGES_PATH,
                        SEG_IMAGES_PATH, 
                        states_log, 
                        actions_log, 
                        font, 
                        screen, 
                        clock, 
                        states_image_log, 
                        image_queue_dis,
                        data_image_dis_queue,
                        data_image_sav_queue,
                        seg_image_dis_queue,
                        seg_image_sav_queue, 
                        max_steps=2000, 
                        frame_id=0):
    step = 0
    destination = carla.Location(x=-76, y=-4, z=0.1)
    # agent = BasicAgent(vehicle)
    agent = BehaviorAgent(vehicle, behavior='normal')  # Using BehaviorAgent for more complex behaviors
    agent.set_destination(destination)
    try:
        while step < max_steps:
            if not vehicle.is_alive:
                print("Vehicle is not alive, stopping data collection.")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame quit event received. Stopping current episode.")
                    return # Exit this function
        
            control = agent.run_step()
            vehicle.apply_control(control)
            world.tick()  # Synchronize the world after applying control
            # done = False
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            states = {
                'location': (transform.location.x, transform.location.y, transform.location.z),
                'rotation': (transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw),
                'velocity': (velocity.x, velocity.y, velocity.z)
            }

            if not image_queue_dis or not data_image_dis_queue or not seg_image_dis_queue:
                clock.tick(20) # Match CARLA FPS
                continue
            surv_image = image_queue_dis[-1]
            data_image = data_image_dis_queue[-1]
            seg_image = seg_image_dis_queue[-1]

            if not isinstance(surv_image, np.ndarray) or surv_image.size == 0:
                print("警告: 监控相机图像无效")
                continue
            if not isinstance(data_image, np.ndarray) or data_image.size == 0:
                print("警告: 数据相机图像无效")
                continue
            if not isinstance(seg_image, np.ndarray) or seg_image.size == 0:
                print("警告: 分割相机图像无效")
                continue

            # Check image dimensions
            if surv_image.ndim != 3 or data_image.ndim != 3 or seg_image.ndim != 3:
                print(f"警告: 图像维度错误 - 监控:{surv_image.shape}, 数据:{data_image.shape}, 分割:{seg_image.shape}")
                continue

            #carla_image = image_queue[-1] # Get the most recent image
            
            # img_bgra = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8")).reshape((800, 720, 4))
            # img_bgr = img_bgra[:, :, :3] # Discard alpha
            # img_rgb_display = img_bgr[:, :, ::-1].copy() # Convert BGR to RGB for display
            # img_to_save = img_bgr # Save as BGR (OpenCV standard) or convert as needed

            # Pygame display
            # surface2 = pygame.image.frombuffer(img_rgb_display.tobytes(), (800, 720), "RGB")
            surface1 = pygame.surfarray.make_surface(surv_image.swapaxes(0, 1))
            surface2 = pygame.surfarray.make_surface(data_image.swapaxes(0, 1))
            seg_surface = pygame.surfarray.make_surface(seg_image.swapaxes(0, 1))
            screen.blit(surface1, (0, 0))
            screen.blit(surface2, (601, 0))
            screen.blit(seg_surface, (1202, 0))

        
            text_surface1 = font.render('Surveillance Camera', True, FULLBLACK, FULLGREEN)
            text_surface1_rect = text_surface1.get_rect()
            text_surface1_rect.center = (400, 60)
            text_surface2 = font.render('Data collector Camera', True, FULLBLACK, FULLRED)
            text_surface2_rect = text_surface2.get_rect()
            text_surface2_rect.center = (1000, 60)
            text_surface3 = font.render('Segmentation Camera', True, FULLBLACK, FULLRED)
            text_surface3_rect = text_surface3.get_rect()
            text_surface3_rect.center = (1600, 60)
            screen.blit(text_surface1, text_surface1_rect)
            screen.blit(text_surface2, text_surface2_rect)
            screen.blit(text_surface3, text_surface3_rect)

            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            actions = [throttle, steer, brake]
            pygame.display.flip()

            # -- Log states and actions ---
            current_timestamp = time.time()  #生成的时间戳形式为浮点数，如1598767892.345678，代表秒数
            image_filename = f"{episode_id}_{frame_id:06d}.png" # Or .jpg，06d表示6位数字，不足的补0
            image_path = os.path.join(IMAGES_PATH, image_filename)
            seg_path = os.path.join(SEG_IMAGES_PATH, image_filename)
            # pygame.image.save(pygame.surfarray.make_surface(img_to_save.swapaxes(0,1)), os.path.join(IMAGES_PATH, image_filename))
            img_to_save = data_image_sav_queue[-1]
            seg_to_save = seg_image_sav_queue[-1]
            origin_data_saving(image_filename,episode_id, current_timestamp, states, actions, states_log, actions_log, image_path, seg_path, img_to_save, seg_to_save)
            data_saving(image_filename, episode_id, current_timestamp, states_image_log, img_to_save)
            frame_id += 1

            clock.tick(20) # Match CARLA FPS
            if agent.done():
                print(f"The agent has reached the destination in episode {episode_id}.")
                break
    except Exception as e:
        print(f"An error occurred during data collection: {e}")
        return

def main():
    # --- Data Storage start ---
    DATA_PATH_AGENT = "agent"
    IMAGES_PATH = os.path.join(DATA_PATH_AGENT, "images")
    SEG_IMAGES_PATH = os.path.join(DATA_PATH_AGENT, "seg_images")
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    if not os.path.exists(SEG_IMAGES_PATH):
        os.makedirs(SEG_IMAGES_PATH)
    states_log = open(os.path.join(DATA_PATH_AGENT, "state.csv"), "w")
    states_log.write("episode_id,timestamp,location_x,location_y,location_z,rotation_roll,rotation_pitch,rotation_yaw,velocity_x,velocity_y,velocity_z\n")
    actions_log = open(os.path.join(DATA_PATH_AGENT, "actions.csv"), "w")
    actions_log.write("episode_id,timestamp,image_file,throttle,steer,brake\n")
    states_image_log = open(os.path.join(DATA_PATH_AGENT, "states_image.csv"), "w")
    states_image_log.write("episode_id,timestamp,image_file,state_image,other_indicators\n")
    # --- Data Storage end ---
    
    FULL_BLACK = (255, 255, 255)
    FULL_GREEN = (0, 255, 0)
    FULL_RED = (255, 0, 0)

    ego_vehicle = None
    surv_camera = None
    data_camera = None
    seg_camera = None
    max_episode_num = 10

    # --- Pygame Setup start ---
    pygame.init()
    screen = pygame.display.set_mode((1803, 541))
    pygame.display.set_caption("Driving Data Collector")
    pygame.font.init()
    font = pygame.font.Font('freesansbold.ttf', 32)

    clock = pygame.time.Clock()
    # --- Pygame Setup end ---

    # --- ego vehicle setup start ---
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('role_name', 'ego')
    vehicle_bp.set_attribute('color', '255,0,0')  # Red color for visibility
    spawn_point = map.get_spawn_points()[0]
    spawn_point.location.x = -53
    spawn_point.location.y = -25
    spawn_point.location.z = 0.1
    spawn_point.rotation.yaw = 90
    # --- ego vehicle setup end ---

    # --- Draw the target point start ---
    target_point = carla.Location()
    target_point.x = -76
    target_point.y = -4
    target_point.z = 0.1
    world.debug.draw_point(target_point, size=0.2, color=carla.Color(255, 0, 0), life_time=1000.0)
    # --- Draw the target point end ---

    for episode_id in range(max_episode_num):
        ## --- reset the world and agent for each episode ---
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # --- set up the cameras ---
        # Surveillance Camera for RGB
        surv_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        
        surv_camera_bp.set_attribute('image_size_x', '600')
        surv_camera_bp.set_attribute('image_size_y', '540')
        surv_camera_transform = carla.Transform(carla.Location(x=-5, y=0, z=5), carla.Rotation(pitch=12, yaw=0, roll=0))
        surv_camera = world.spawn_actor(surv_camera_bp, surv_camera_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
        image_queue_dis = []
        image_queue_sav = []
        # TODO: : 重置回调函数，防止数据丢失
        # surv_camera.listen(lambda image: image_queue_dis.append(process_img(image)[0]) if len(image_queue_dis) < 2 else image_queue_dis.pop(0))
        surv_camera.listen(lambda image: try_add_to_queue(image_queue_dis, image_queue_sav, process_img(image)[0], process_img(image)[1]))

        # Camera Sensor for data collection
        data_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        data_camera_bp.set_attribute('image_size_x', '600')
        data_camera_bp.set_attribute('image_size_y', '540')
        data_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        data_camera = world.spawn_actor(data_camera_bp, data_camera_transform, attach_to=ego_vehicle)
        data_image_dis_queue = [] 
        data_image_sav_queue = []
        # data_camera.listen(lambda image: data_image_dis_queue.append(process_img(image)[0]) if len(data_image_dis_queue) < 2 else data_image_dis_queue.pop(0))
        data_camera.listen(lambda image: try_add_to_queue(data_image_dis_queue, data_image_sav_queue, process_img(image)[0], process_img(image)[1]))

        seg_camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '600')
        seg_camera_bp.set_attribute('image_size_y', '540')
        seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=3.0))
        seg_camera = world.spawn_actor(seg_camera_bp, seg_camera_transform, attach_to=ego_vehicle)
        seg_image_dis_queue = [] 
        seg_image_sav_queue = []
        # seg_camera.listen(lambda image: seg_image_dis_queue.append(process_seg_img(image)[0]) if len(seg_image_dis_queue) < 2 else seg_image_dis_queue.pop(0))
        seg_camera.listen(lambda image: try_add_to_queue(seg_image_dis_queue, seg_image_sav_queue, process_seg_img(image)[0], process_seg_img(image)[1]))

        ## --- set up the agent ---
        if ego_vehicle is None:
            print("Error: ego_vehicle is None. Skipping agent setup for this episode.")
            continue
        agent_e = BehaviorAgent(ego_vehicle, behavior='normal')  # Using BehaviorAgent for more complex behaviors
        world.tick()  # Synchronize the world to apply the agent's destination

        ## --- run the data collection ---
        run_collection_data(world=world, 
                            agent=agent_e, 
                            vehicle=ego_vehicle, 
                            episode_id=episode_id, 
                            IMAGES_PATH=IMAGES_PATH,
                            SEG_IMAGES_PATH=SEG_IMAGES_PATH, 
                            states_log=states_log, 
                            actions_log=actions_log, 
                            font=font, screen=screen, 
                            clock=clock, 
                            states_image_log=states_image_log, 
                            image_queue_dis=image_queue_dis,
                            data_image_dis_queue=data_image_dis_queue,
                            data_image_sav_queue=data_image_sav_queue,
                            seg_image_dis_queue=seg_image_dis_queue,
                            seg_image_sav_queue=seg_image_sav_queue,
                            max_steps=2000, 
                            frame_id=0)

        ## --- Clean up after each episode ---
        if ego_vehicle is not None:
            data_camera.stop()
            surv_camera.stop()
            seg_camera.stop()
            data_camera.destroy()
            surv_camera.destroy()
            seg_camera.destroy()
            ego_vehicle.destroy()
            data_camera = None
            surv_camera = None
            seg_camera = None
            ego_vehicle = None
        if data_camera is not None:
            data_camera.stop()
            data_camera.destroy()
            data_camera = None
        if surv_camera is not None:
            surv_camera.stop()
            surv_camera.destroy()
            surv_camera = None
        if seg_camera is not None:
            seg_camera.stop()
            seg_camera.destroy()
            seg_camera = None
        time.sleep(1.0)  # Wait a bit before starting the next episode
    # --- Clean up ---
    if surv_camera is not None:
        surv_camera.destroy()
    if data_camera is not None:
        data_camera.destroy()
    if seg_camera is not None:
        seg_camera.destroy()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    print("Data collection completed.")

def process_img(image):
    #image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    # array = np.array(image.raw_data)
    array = np.reshape(array, (image.height, image.width, 4))
    array_sav = array[:, :, :3]
    array_dis = array[:, :, :3][:, :, ::-1]  # BGR to RGB
    rgb_image_display = array_dis.copy()
    rgb_image_saving = array_sav.copy()
    return rgb_image_display, rgb_image_saving

def process_seg_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array_sav = array[:, :, :3]
    array_dis = array_sav[:, :, ::-1]  # BGR to RGB

    seg_image_display = array_dis.copy()
    seg_image_saving = array_sav.copy()
    return seg_image_display, seg_image_saving

def try_add_to_queue(queue_dis, queue_sav, item_dis, item_sav, max_size=2):
    """
    Try to add an item to the queue. If the queue is full, remove the oldest item.
    """
    if len(queue_dis) < max_size:
        queue_dis.append(item_dis)
    else:
        queue_dis.pop(0)
        queue_dis.append(item_dis)

    if len(queue_sav) < max_size:
        queue_sav.append(item_sav)
    else:
        queue_sav.pop(0)
        queue_sav.append(item_sav)


def origin_data_saving(image_filename, episode_id, current_timestamp, states, actions, states_log, actions_log, image_path, seg_path, img_to_save, seg_to_save):
    cv2.imwrite(image_path, img_to_save)
    cv2.imwrite(seg_path, seg_to_save)
    throttle = actions[0]
    steer = actions[1]
    brake = actions[2]
    x = states['location'][0]
    y = states['location'][1]
    z = states['location'][2]
    pitch = states['rotation'][0]
    yaw = states['rotation'][1]
    roll = states['rotation'][2]
    velocity_x = states['velocity'][0]
    velocity_y = states['velocity'][1]
    velocity_z = states['velocity'][2]
    states_log.write(f"{episode_id},{current_timestamp},{x},{y},{z},{pitch},{yaw},{roll},{velocity_x},{velocity_y},{velocity_z}\n")
    actions_log.write(f"{episode_id},{current_timestamp},{image_filename},{throttle},{steer},{brake}\n")

def data_saving(image_filename, episode_id, current_timestamp, states_image_log, img_to_save):
    # TODO: :增加语义图像的保存逻辑
    state_image = img_to_save[:, :, 0]
    state_image = cv2.resize(state_image, (50, 45))
    state_image = np.float16(np.squeeze(state_image)/ 255.0)  # np.squeeze将去掉单维度，转换为float16类型并归一化到0-1范围
    state_image = state_image.tolist() # 将numpy数组转换为列表

    other_indicators = None
    states_image_log.write(f"{episode_id},{current_timestamp},{image_filename},{state_image},{other_indicators}\n")
    

if __name__ == '__main__':
    main()