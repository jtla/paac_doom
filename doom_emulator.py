import numpy as np
#from ale_python_interface import ALEInterface
from scipy.misc import imresize
from cv2 import cvtColor, COLOR_RGB2GRAY
import random
from environment import BaseEnvironment, FramePool,ObservationPool

from vizdoom import *
#IMG_SIZE_X = 84
#IMG_SIZE_Y = 84

IMG_SIZE_X = 160
IMG_SIZE_Y = 120

NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2


class DoomEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        #self.ale = ALEInterface()
        self.doom = DoomGame()

        self.doom.set_doom_scenario_path("scenarios/basic.wad")
        self.doom.set_doom_map("map01")

        #self.ale.setInt(b"random_seed", args.random_seed * (actor_id +1))
        self.doom.set_seed(args.random_seed * (actor_id +1))

        self.doom.set_screen_resolution(ScreenResolution.RES_160X120)
        #self.doom.set_screen_format(ScreenFormat.CRCGCB)
        #self.doom.set_screen_resolution(ScreenResolution.RES_640X480)
        self.doom.set_screen_format(ScreenFormat.RGB24)

        # Enables depth buffer.
        self.doom.set_depth_buffer_enabled(True)
        #self.doom.set_depth_buffer_enabled(False)

        self.doom.set_labels_buffer_enabled(False)
        self.doom.set_automap_buffer_enabled(False)

        self.doom.set_render_hud(False)
        self.doom.set_render_minimal_hud(False)  # If hud is enabled
        self.doom.set_render_crosshair(False)
        self.doom.set_render_weapon(True)
        self.doom.set_render_decals(False)  # Bullet holes and blood on the walls
        self.doom.set_render_particles(False)
        self.doom.set_render_effects_sprites(False)  # Smoke and blood
        self.doom.set_render_messages(False)  # In-game messages
        self.doom.set_render_corpses(False)
        self.doom.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

        # Adds buttons that will be allowed.
        self.doom.add_available_button(Button.MOVE_LEFT)
        self.doom.add_available_button(Button.MOVE_RIGHT)
        self.doom.add_available_button(Button.ATTACK)

        # Adds game variables that will be included in state.
        self.doom.add_available_game_variable(GameVariable.AMMO2)

        # Causes episodes to finish after 200 tics (actions)
        self.doom.set_episode_timeout(201)

        # Makes the window appear (turned on by default)
        self.doom.set_window_visible(False)

        # Turns on the sound. (turned off by default)
        self.doom.set_sound_enabled(False)

        # Sets the livin reward (for each move) to -1
        self.doom.set_living_reward(-1)

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.doom.set_mode(Mode.PLAYER)

        # Initialize the game. Further configuration won't take any effect from now on.
        # self.doom.init()

#        import pdb;pdb.set_trace()

        # For fuller control on explicit action repeat (>= ALE 0.5.0)
        #self.ale.setFloat(b"repeat_action_probability", 0.0)
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj

        #self.ale.setInt(b"frame_skip", 1)
        # frame_skip = 1

        #self.ale.setBool(b"color_averaging", False)
        #full_rom_path = args.rom_path + "/" + args.game + ".bin"

        #self.ale.loadROM(str.encode(full_rom_path))

        #self.legal_actions = self.ale.getMinimalActionSet()
        # self.doom.getActions()

        # actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.legal_actions = [[True, False, False], [False, True, False], [False, False, True]]

        #self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.screen_width = self.doom.get_screen_width()
        self.screen_height = self.doom.get_screen_height()

        #self.lives = self.ale.lives()

        # parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
        self.random_start = args.random_start
        # Makes episodes start after 10 tics (~after raising the weapon)
        # self.doom.set_episode_start_time(10)
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            self.doom.set_episode_start_time(wait)

        # parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
        self.single_life_episodes = args.single_life_episodes
        # parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
        self.call_on_new_frame = args.visualize

        # Processed historical frames that will be fed in to the network
        # (i.e., four 84x84 images)
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8))
        self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height, self.screen_width,1), dtype=np.uint8)
        self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width), dtype=np.uint8),
                                    self.__process_frame_pool)

        self.doom.init()

        self.debug_counter = 0
        self.debug_counter2 = 0
        self.debug_counter3 = 0
        self.debug_counter4 = 0

        #for testing purposes
        #self.doom.init()
        #self.doom.new_episode()
        #state = self.doom.get_state()
        #screen = state.screen_buffer
        #depth = state.screen_buffer
        #import cv2
        #cv2.imwrite('asdf.bmp', depth)
        #import pdb;pdb.set_trace()

    def get_legal_actions(self):
        return self.legal_actions

    def __get_screen_image(self):
        """
        Get the current frame luminance
        :return: the current frame
        """
        state = self.doom.get_state()
        #self.debug_counter += 1
        #print("{} {} {}".format(self.debug_counter, state, self.doom.is_episode_finished()))
        #import pdb;pdb.set_trace()
        self.rgb_screen = state.screen_buffer
        self.gray_screen = cvtColor(self.rgb_screen, COLOR_RGB2GRAY)
        if self.call_on_new_frame:
            self.on_new_frame(self.rgb_screen)
        return np.squeeze(self.gray_screen)

        #self.ale.getScreenGrayscale(self.gray_screen)
        #if self.call_on_new_frame:
        #    self.ale.getScreenRGB(self.rgb_screen)
        #    self.on_new_frame(self.rgb_screen)
        #return np.squeeze(self.gray_screen)

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        #print("__new_game")
        self.doom.new_episode()

        #self.ale.reset_game()
        #self.lives = self.ale.lives()
        #if self.random_start:
        #    wait = random.randint(0, MAX_START_WAIT)
        #    for _ in range(wait):
        #        self.ale.act(self.legal_actions[0])

    # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    # There’s one more obscure step that Google DeepMind did: they took the component-wise maximum over two consecutive frames,
    # which helps DQN deal with the problem of how certain Atari games only render their sprites every other game frame
    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool """

        img = np.amax(frame_pool, axis=0)
        #img = imresize(img, (84, 84), interp='nearest')
        img = imresize(img, (160, 120), interp='nearest')
        img = img.astype(np.uint8)
        #import pdb;pdb.set_trace()
        return img


    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            #reward += self.ale.act(self.legal_actions[a])
            #self.debug_counter4 += 1
            #print("Debug_4 {} {}".format(self.debug_counter4, self.doom.is_episode_finished()))
            #self.debug_counter4 += 1
            #print("Debug_4 {} {}".format(self.debug_counter4, self.doom.is_episode_finished()))
            reward += self.doom.make_action(self.legal_actions[a])
            if self.__is_terminal():
                return reward

        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            #reward += self.ale.act(self.legal_actions[a])
            #self.debug_counter3 += 1
            #print("Debug_3_1 {} {}".format(self.debug_counter3, self.doom.is_episode_finished()))
            #print("Debug_3_1 {} {}".format(self.debug_counter3, self.doom.is_episode_finished()))
            reward += self.doom.make_action(self.legal_actions[a])
            if self.__is_terminal():
                return reward
            #print("Debug_3 {} {}".format(self.debug_counter3, self.doom.is_episode_finished()))
            self.frame_pool.new_frame(self.__get_screen_image())
        return reward

    def get_initial_state(self):
        """ Get the initial state """
        self.__new_game()
        for step in range(NR_IMAGES):
            _ = self.__action_repeat(0)
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if self.__is_terminal():
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        reward = self.__action_repeat(np.argmax(action))
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        terminal = self.__is_terminal()
        observation = self.observation_pool.get_pooled_observations()
        return observation, reward, terminal

        #reward = self.__action_repeat(np.argmax(action))
        #self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        #terminal = self.__is_terminal()
        #self.lives = self.ale.lives()
        #observation = self.observation_pool.get_pooled_observations()
        #return observation, reward, terminal

    def __is_terminal(self):
        #self.debug_counter2 += 1
        #print("{} {}".format(self.debug_counter2, self.doom.is_episode_finished()))
        return self.doom.is_episode_finished()
        #return self.__is_over()

        #if self.single_life_episodes:
        #    return self.__is_over() or (self.lives > self.ale.lives())
        #    return self.__is_over() or (self.lives > self.doom.lives())
        #else:
        #    return self.__is_over()

    #def __is_over(self):
        #self.debug_counter2 += 1
        #print("{} {}".format(self.debug_counter2, self.doom.is_episode_finished()))
        #return self.doom.is_episode_finished()

        #return self.ale.game_over()

    #https://github.com/mwydmuch/ViZDoom/issues/71
    #Noop is actually setting all buttons to False/0 sousing e.g. 3 buttons to perform Noop use:
    #make_action([0,0,0]) / make_action([False,False,False]). If you are lazy you can use an empty list since it will be filled with required number of 0 so:
    #make_action([]) is a NoOp.
    def get_noop(self):
        return [0, 0, 0]

        #return [1.0, 0.0]
