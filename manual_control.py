#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# for human demo
import json 
import matplotlib.pyplot as plt
import cv2
import h5py
import os

class ManualControl:
    def __init__(
        self,
        env: Env,
        args,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.args = args
        

    def start(self):

        if args.con:

            # continoued recording
            with open('human_demo_data/hd_minigrid.json', 'r') as file:
                self.hd_data = json.load(file)

            self.c_target = None
            self.c_action = []
            self.c_obs = []
            self.c_index = len(self.hd_data["episodes"])

        else:
            # start fresh recording
            self.hd_data = {"episodes":[]}
            self.c_target = None
            self.c_action = []
            self.c_obs = []
            self.c_index = 0

        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)


    def record_current(self):
        self.hd_data["episodes"].append({"index":self.c_index,"language":self.c_target,"actions":self.c_action})

        imgfolder = "human_demo_data/" + str(self.c_index) +"_imgSeq"
        os.makedirs(imgfolder)

        # save image sequence
        with h5py.File(imgfolder+"/imgSeq", 'w') as hdf5_file:
            hdf5_file.create_dataset("imgSeq", data=self.c_obs)

        with open('human_demo_data/hd_minigrid.json', 'w') as file:
            json.dump(self.hd_data, file, indent=4)

        print(len(self.c_action))
        print(len(self.c_obs))
        # reset recordings

        self.c_target = None
        self.c_action = []
        self.c_obs = []

        self.c_index += 1


    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        # print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.c_obs.append(obs["image"])
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.c_obs.append(obs["image"])
            self.env.render()

    def reset(self, seed=None):
        
        if self.c_target != None:
            self.record_current()

        obs,info = self.env.reset(seed=seed)
        print(obs["mission"])
        print(obs["image"].shape)
        self.c_obs.append(obs["image"])
        self.c_target = obs["mission"]

        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        # print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "return": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.c_action.append(action)
            self.step(action)
            
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="BabyAI-GoToSeqS5R2-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )

    parser.add_argument(
        "--con",
        action="store_true",
        help="continou recording.",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=True,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    env = RGBImgPartialObsWrapper(env, args.tile_size)
    
    manual_control = ManualControl(env, args,seed=args.seed)
    manual_control.start()
