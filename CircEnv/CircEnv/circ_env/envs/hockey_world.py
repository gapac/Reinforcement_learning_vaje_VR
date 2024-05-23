import gym
from gym import spaces
import numpy as np
import math
import pygame
from pygame.locals import *
from Box2D import *
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody) 
from circ_env.envs.line import Line
from circ_env.envs.dimensions import Dimensions
from circ_env.envs.borders import generateBorder
from circ_env.envs.ai import AI

PLAYER_CATEGORY = 0x0001
PUCK_CATEGORY = 0x0002
ALL_CATEGORY = 0xFFFF
PLAYER_MASK = PUCK_CATEGORY | ALL_CATEGORY
PUCK_MASK = PLAYER_CATEGORY | ALL_CATEGORY

PPM = 100.0  # pixels per m
TARGET_FPS = 50
TIME_STEP = 1.0 / TARGET_FPS
WORLD_WIDTH, WORLD_HEIGHT = 450/PPM, 800/PPM

class ContactListener(b2ContactListener):
    def __init__(self):
        super().__init__()
        self._contact = False

    def BeginContact(self, contact):
        self._contact = True

    def EndContact(self, contact):
        self._contact = False

    def contact(self):
        return self._contact
    
    # def ShouldCollide(self, fixtureA, fixtureB):
    #     if fixtureA.categoryBits == PLAYER_CATEGORY and fixtureB.categoryBits == PUCK_CATEGORY:
    #         return False
    #     elif fixtureA.categoryBits == PUCK_CATEGORY and fixtureB.categoryBits == PLAYER_CATEGORY:
    #         return False
    #     else:
    #         return True   

class HockeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": TARGET_FPS}

    def __init__(self, render_mode=None):
        super(HockeyEnv, self).__init__()
        # Set the environment parameters
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT
        self.PPM = PPM
        self.current_step = 0
        # Initialize the Pybox2D world
        self.world = b2World(gravity=(0, 0))        

        self.dim = Dimensions()

        self.score_agent = 0
        self.score_opponent = 0
        ############ TUKAJ SPREMINJATE

        player_radius_px = 30
        object_radius_px = 30
        self.target_radius = 75/self.PPM 
        self.player_max_vel = 5.0
        self.puck_max_vel = 10.0*np.sqrt(2)

        self.time_steps = 500

        # Set the observation and action spaces
        #self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -np.finfo(np.float32).max, -np.finfo(np.float32).max]), high=np.array([self.width, self.height, self.width, self.height, np.finfo(np.float32).max, np.finfo(np.float32).max]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0,     -self.player_max_vel,   -self.player_max_vel, 
                                                          0.0, 0.0,     -self.player_max_vel,   -self.player_max_vel, 
                                                          0.0, 0.0,     -self.puck_max_vel,     -self.puck_max_vel]), 
                                            high=np.array([self.width, self.height, self.player_max_vel,   self.player_max_vel,
                                                           self.width, self.height, self.player_max_vel,   self.player_max_vel,
                                                           self.width, self.height, self.puck_max_vel,   self.puck_max_vel]), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.player_max_vel, high=self.player_max_vel, shape=(2,), dtype=np.float32)
        
        self.create_border()
        self.create_hockey_border()
        self.create_central_line_border()

        self.create_agent(player_radius_px, 0.1)
        self.create_opponent(player_radius_px, 0.1)
        self.create_puck(object_radius_px, 0.05, 'd')

        self.create_goal_top((150, 10), b2Vec2(self.dim.center[0]/self.PPM, (self.dim.rink_top-10)/self.PPM))
        self.create_goal_bottom((150, 10), b2Vec2(self.dim.center[0]/self.PPM, (self.dim.rink_bottom+10)/self.PPM))

        self.top_ai    = AI(self.opponent,    self.object, mode='bottom',    dim=self.dim)
        self.bottom_ai = AI(self.agent,    self.object, mode='top',    dim=self.dim)

        self.target = b2Vec2(self.dim.center[0]/self.PPM, self.dim.rink_bottom/self.PPM-self.target_radius)

        ############ DO TUKAJ SPREMINJATE
        # Set the contact listener
        self.contact_listener = ContactListener()
        self.world.contactListener = self.contact_listener

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None     

    def _get_obs(self):
        agent_pos = self.get_agent_position()
        agent_vel = self.get_agent_velocity()
        opp_pos = self.get_opponent_position()
        opp_vel = self.get_opponent_velocity()
        object_pos = self.get_puck_position()
        object_vel = self.get_puck_velocity()
        #target_pos = self.get_target_position()
        
        return np.concatenate((agent_pos, agent_vel, opp_pos, opp_vel, object_pos, object_vel), axis=0)
    
    def reset(self):
        self.world.ClearForces()
        # Reset the current step counter
        self.current_step = 0
        
        self.reset_agent()
        self.reset_opponent()

        self.reset_puck()
        
       # Return the initial observation
        return self._get_obs()
    
    def step(self, action):

        ############ TUKAJ SPREMINJATE
        action = self.bottom_ai.move()
        #action = self.move_agent_mouse()
        self.set_agent_velocity(action)

        robot_action = self.top_ai.move()
        self.set_opponent_velocity(robot_action)

        self.limit_puck_velocity(self.puck_max_vel)
     
        # Get the current observation, reward, and done flag
        obs = self._get_obs()
        
        reward = 0
        done = False        

        # Check if player (bottom mallet) scored a goal
        if self._is_collision(self.object, self.goal_t):
            self.score_agent += 1
            reward = 1.0
            done = True

        # Check if opponent (top mallet) scored a goal
        if self._is_collision(self.object, self.goal_b):
            self.score_opponent += 1
            reward = -1.0
            done = True            
        
        # Check if episode is too long
        if self.current_step >= self.time_steps:
            reward = -1.0
            done = True

        # Check if player hit the border
        if self._is_collision(self.agent, self.border):
            reward = -0.1
            done = True            

        ############ DO TUKAJ SPREMINJATE
        self.current_step += 1
        self.world.Step(TIME_STEP, 6, 2)
        self.world.ClearForces()

        return obs, reward, done, {}       

    def render(self, render_mode):
        if self.render_mode == "human":
            return self._render_frame()


    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width*self.PPM, self.height*self.PPM))
            pygame.display.set_caption("Hockey Environment")
            self.font = pygame.font.SysFont("monospace", 30)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Define colors
        white = (255, 255, 240)
        black = (0, 0, 0)
        red = (202, 52, 51)
        green = (76, 187, 23)
        light_green = (194, 218, 184)
        blue = (0, 71, 171)
        yellow = (219,195,0)
        
        self.screen.fill(white)

        ############ TUKAJ SPREMINJATE
        # Draw the boundaries
        self.draw_border(black)
        self.draw_hockey_border(black)
        self.draw_central_line(blue)

        self.draw_target(light_green)
        self.draw_agent(green)
        self.draw_opponent(red)
        self.draw_puck(yellow)
       
        self.draw_goal_top(blue)
        self.draw_goal_bottom(blue)


        ############ DO TUKAJ SPREMINJATE

        self.screen.blit(self.font.render('%4d' % self.score_opponent,    1, red), (0, 10))
        self.screen.blit(self.font.render('%4d' % self.score_agent, 1, green), (0, self.height*self.PPM-40))

        if self.render_mode == "human":
            # Update the Pygame display
            pygame.display.flip()
            pygame.display.update()
            pygame.event.pump()
            pygame.event.clear()
            self.clock.tick(self.metadata["render_fps"])

        
    def close(self):
        self.world.DestroyBody(self.agent)
        self.world.DestroyBody(self.object)
        self.world.DestroyBody(self.opponent)
        self.world.DestroyBody(self.goal_t)
        self.world.DestroyBody(self.goal_b)
        self.world.contactListener = None
        self.world = None
        
        # Quit Pygame
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _is_collision(self, object, goal):
        for fixture_object in object.fixtures:
            for fixture_goal in goal.fixtures:
                if b2TestOverlap(fixture_object.shape, 0, fixture_goal.shape, 0, object.transform, goal.transform):
                    return True
        return False   

    def create_opponent(self, radius_px, friction):
        self.opponent_radius = radius_px/self.PPM
        density = 1.0
        friction = friction
        restitution = 0.5
        opponent_fixture = b2FixtureDef(shape=b2CircleShape(radius=self.opponent_radius), density=density, friction=friction, restitution=restitution)
        opponent_fixture.filter.categoryBits = PLAYER_CATEGORY
        opponent_fixture.filter.maskBits = PLAYER_MASK 
        self.opponent = self.world.CreateDynamicBody(position=(0,0), fixtures=opponent_fixture)
        self.reset_opponent()

    def create_agent(self, radius_px, friction):
        self.agent_radius = radius_px/self.PPM
        density = 1.0
        friction = friction
        restitution = 0.5        
        agent_fixture = b2FixtureDef(shape=b2CircleShape(radius=self.agent_radius), density=density, friction=friction, restitution=restitution)
        agent_fixture.filter.categoryBits = PLAYER_CATEGORY
        agent_fixture.filter.maskBits = PLAYER_MASK  
        self.agent = self.world.CreateDynamicBody(position=(0,0), fixtures=agent_fixture)
        self.reset_agent()

    def create_puck(self, radius_px, friction, type):
        self.object_radius = radius_px/self.PPM
        self.object_density = 0.1
        self.object_friction = friction
        self.object_restitution = 0.8
        shape = b2CircleShape(radius=self.object_radius)
        object_fixture = b2FixtureDef(shape=shape, density=self.object_density, friction=friction, restitution=self.object_restitution)
        object_fixture.filter.categoryBits = PUCK_CATEGORY
        object_fixture.filter.maskBits = PUCK_MASK      
        if (type == 'd'):
            self.object = self.world.CreateDynamicBody(position=(0,0), fixtures=object_fixture)
        if (type == 'k'):
            self.object = self.world.CreateKinematicBody(position=(0,0), fixtures=object_fixture)
        self.reset_puck()

    def create_goal_top(self, dim_px, position):
        goal_width, goal_height = dim_px[0]/self.PPM, dim_px[1]/self.PPM
        self.goal_t_fixture = b2FixtureDef(shape=b2PolygonShape(box=(goal_width/2, goal_height/2)), density=0.0, friction=0.5, restitution=0, isSensor=True)
        self.goal_t = self.world.CreateStaticBody(position=position, fixtures=self.goal_t_fixture)

    def create_goal_bottom(self, dim_px, position):
        goal_width, goal_height = dim_px[0]/self.PPM, dim_px[1]/self.PPM
        self.goal_b_fixture = b2FixtureDef(shape=b2PolygonShape(box=(goal_width/2, goal_height/2)), density=0.0, friction=0.5, restitution=0, isSensor=True)
        self.goal_b = self.world.CreateStaticBody(position=position, fixtures=self.goal_b_fixture)

    def create_circ_target(self, radius_px):
        self.target_radius = radius_px/self.PPM
        self.target_radius = radius_px/self.PPM
        self.target = b2Vec2(np.random.uniform(self.agent_radius*2, self.width - self.agent_radius*2), np.random.uniform(self.agent_radius*2, self.height - self.agent_radius*2))

    def create_border(self):
        self.border = self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(0.0, 0.0), (self.width, 0.0)]),
                b2EdgeShape(vertices=[(0.0, self.height), (self.width, self.height)]),
                b2EdgeShape(vertices=[(0.0, 0.0), (0.0, self.height)]),
                b2EdgeShape(vertices=[(self.width, 0.0), (self.width, self.height)])
            ]
        ) 
        
    def create_hockey_border(self, dim=Dimensions()):
        self.dim = dim
        # Add Walls
        # Arcs and Lines have to be passed in an anti-clockwise order with respect to the self.dim.center
        top_wall          = Line(self.dim.arc_top_left_start, self.dim.arc_top_right_end)
        bottom_wall       = Line(self.dim.arc_bottom_left_end, self.dim.arc_bottom_right_start)
        left_wall         = Line(self.dim.arc_top_left_end, self.dim.arc_bottom_left_start)
        right_wall        = Line(self.dim.arc_top_right_start, self.dim.arc_bottom_right_end)

        top_left_wall     = Line(self.dim.arc_top_left_start, self.dim.post_top_left)
        top_right_wall    = Line(self.dim.post_top_right, self.dim.arc_top_right_end)
        bottom_left_wall  = Line(self.dim.arc_bottom_left_end, self.dim.post_bottom_left)
        bottom_right_wall = Line(self.dim.post_bottom_right, self.dim.arc_bottom_right_start)

        center_line       = Line(self.dim.center_left, self.dim.center_right)
        
        # Add Corners
        top_left_corner     = Line.generate_bezier_curve(self.dim.arc_top_left, self.dim.bezier_ratio)
        top_right_corner    = Line.generate_bezier_curve(self.dim.arc_top_right, self.dim.bezier_ratio)
        bottom_left_corner  = Line.generate_bezier_curve(self.dim.arc_bottom_left, self.dim.bezier_ratio)
        bottom_right_corner = Line.generate_bezier_curve(self.dim.arc_bottom_right, self.dim.bezier_ratio)

        #borders
        self.border_l = [top_left_wall] + top_left_corner + [left_wall] + bottom_left_corner + [bottom_left_wall]# + [center_line]
        self.border_r = [top_right_wall] + top_right_corner + [right_wall] + bottom_right_corner + [bottom_right_wall]

        shape_border_l_body = generateBorder(self.border_l)
        shape_border_r_body = generateBorder(self.border_r)  
        
        self.hockey_border = self.world.CreateStaticBody(shapes=shape_border_l_body+shape_border_r_body)  

    def create_central_line_border(self):
        self.shape_center_line_vertices = [(self.dim.rink_left/self.PPM,  self.dim.height//2/self.PPM), (self.dim.rink_right/self.PPM, self.dim.height//2/self.PPM)]
        shape_center_line = b2EdgeShape(vertices=self.shape_center_line_vertices)
        fixture_center_line = b2FixtureDef(shape=shape_center_line)
        fixture_center_line.filter.categoryBits = PLAYER_CATEGORY
        fixture_center_line.filter.maskBits = PLAYER_CATEGORY       
        self.center_line_border_body = self.world.CreateStaticBody(fixtures=fixture_center_line)

    def get_agent_position(self):
        return np.array([self.agent.position.x, self.agent.position.y])

    def get_agent_velocity(self):
        return np.array([self.agent.linearVelocity.x, self.agent.linearVelocity.y])
    
    def get_opponent_position(self):
        return np.array([self.opponent.position.x, self.opponent.position.y])

    def get_opponent_velocity(self):
        return np.array([self.opponent.linearVelocity.x, self.opponent.linearVelocity.y])        
    
    def get_puck_position(self):
        return np.array([self.object.position.x, self.object.position.y])
    
    def get_puck_velocity(self):
        return np.array([self.object.linearVelocity.x, self.object.linearVelocity.y])
    
    def get_target_position(self):
        return np.array([self.target.x, self.target.y])
    
    def reset_agent(self):
        self.agent.position = self.random_position(self.agent_radius, self.dim.center[1], self.dim.rink_bottom)
        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0.0

    def reset_opponent(self):
        self.opponent.position = self.random_position(self.opponent_radius, self.dim.rink_top, self.dim.center[1])
        self.opponent.linearVelocity = (0, 0)
        self.opponent.angularVelocity = 0.0

    def reset_puck(self):
        self.object.position = self.random_position(self.object_radius*2, self.dim.rink_top, self.dim.rink_bottom)
        self.object.linearVelocity = (np.random.uniform(-self.puck_max_vel*0.5, self.puck_max_vel*0.5),
                np.random.uniform(-self.puck_max_vel*0.5, self.puck_max_vel*0.5))
        self.object.angularVelocity = 0.0

    def reset_target(self, position):
        self.target.x = position[0]
        self.target.y = position[1]

    def set_agent_velocity(self, vel):
        self.agent.linearVelocity = b2Vec2(vel[0]*1.0, vel[1]*1.0)
        self.agent.angularVelocity = 0.0

    def set_opponent_velocity(self, vel):
        self.opponent.linearVelocity = b2Vec2(vel[0]*1.0, vel[1]*1.0)
        self.opponent.angularVelocity = 0.0        
 
    def set_puck_velocity(self, vel):
        self.object.linearVelocity = b2Vec2(vel[0]*1.0, vel[1]*1.0)
        self.object.angularVelocity = 0.0

    def limit_puck_velocity(self, maximal_velocity):
        # Check if the magnitude of velocity is larger than maximal_velocity
        velocity = self.object.linearVelocity
        if velocity.length > maximal_velocity:
            # If it is, calculate the unit vector of velocity and multiply it by maximal_velocity
            unit_velocity = velocity / velocity.length
            limited_velocity = unit_velocity * maximal_velocity
        else:
            # If it's not, return the original velocity
            limited_velocity = velocity
        self.object.linearVelocity = limited_velocity
        self.object.angularVelocity = 0.0    
  
    def draw_agent(self, color):
        pygame.draw.circle(self.screen, color, (int(self.agent.position.x*self.PPM), int(self.agent.position.y*self.PPM)), int(self.agent_radius*self.PPM))

    def draw_opponent(self, color):
        pygame.draw.circle(self.screen, color, (int(self.opponent.position.x*self.PPM), int(self.opponent.position.y*self.PPM)), int(self.opponent_radius*self.PPM))

    def draw_puck(self, color):
        pygame.draw.circle(self.screen, color, (int(self.object.position.x*self.PPM), int(self.object.position.y*self.PPM)), int(self.object_radius*self.PPM))

    def draw_border(self, color):
        pygame.draw.rect(self.screen, color, (1, 1, int(self.width*self.PPM-1), int(self.height*self.PPM-1)), 1)

    def draw_target(self, color):
        pygame.draw.circle(self.screen, color, (int(self.target.x*self.PPM), int(self.target.y*self.PPM)), int(self.target_radius*self.PPM))

    def draw_goal_top(self, color):
        goal_position = self.goal_t.position
        goal_dimensions = self.goal_t_fixture.shape.vertices
        goal_vertices = [(goal_position[0]*self.PPM + vertex[0]*self.PPM, goal_position[1]*self.PPM + vertex[1]*self.PPM) for vertex in goal_dimensions]
        pygame.draw.polygon(self.screen, color, goal_vertices, 0)

    def draw_goal_bottom(self, color):
        goal_position = self.goal_b.position
        goal_dimensions = self.goal_b_fixture.shape.vertices
        goal_vertices = [(goal_position[0]*self.PPM + vertex[0]*self.PPM, goal_position[1]*self.PPM + vertex[1]*self.PPM) for vertex in goal_dimensions]
        pygame.draw.polygon(self.screen, color, goal_vertices, 0)

    def draw_hockey_border(self, color):
        for line in self.border_l + self.border_r:
            pygame.draw.line(self.screen, color, line.p2, line.p1, 6)  
        
    def draw_central_line(self, color):
        pygame.draw.lines(self.screen, color, False, [(v[0]*self.PPM, v[1]*self.PPM) for v in self.shape_center_line_vertices], 4) 


    def is_agent_outside_screen(self):
        outside = False
        if (self.agent.position.x < self.agent_radius) or (self.agent.position.x > (self.width - self.agent_radius)):     
            outside = True
        if (self.agent.position.y < self.agent_radius) or (self.agent.position.y > (self.height - self.agent_radius)):
            outside = True
        return outside

    def calc_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)
        
    def unit_vector(self, pos1, pos2):
        direction = pos2 - pos1
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            unit_direction = direction / magnitude
        else:
            unit_direction = np.zeros_like(direction)
        return unit_direction

    def calculate_component(self, pos_agent, pos_target, vel):
        # Calculate the unit vector pointing from pos1 to pos2
        direction = pos_target - pos_agent
        unit_vector = direction / np.linalg.norm(direction)

        # Calculate the component of vel in the direction of the unit vector
        component = np.dot(vel, unit_vector)

        # Return the scaled component
        return component

    def random_position(self, radius, top, bottom):
        return (
                np.random.uniform(
                        self.dim.rink_left/self.PPM + radius,
                        self.dim.rink_right/self.PPM - radius
                        ),
                np.random.uniform(
                        top/self.PPM + radius,
                        bottom/self.PPM - radius
                        ))
    
    def move_agent_mouse(self):
        x, y = 0, 0

        if pygame.mouse.get_pressed()[0]:
            try:
                mouse_pos = pygame.mouse.get_pos()
                agent_pos = self.get_agent_position()
                dx = agent_pos[0] - mouse_pos[0]/self.PPM
                dy = agent_pos[1] - mouse_pos[1]/self.PPM
                x, y = -dx*2, -dy*2
            except AttributeError:
                pass
 
        action = np.array([x,y],dtype=np.float32)

        return action          