import math
import os
import cv2
import pygame
import numpy as np
from pygame.locals import *

FPS = 60
ANIMATION_SPEED = 0.18
WIN_WIDTH = 223 * 2
WIN_HEIGHT = 382

# define reward
REWARD_SURVIVE = 1.0
REWARD_PASS = 0.1
REWARD_DEAD = 0.0

# define actions
ACTION_DO_NOTHING = 0
ACTION_FLAP = 1

# define state type
STATE_TYPE_NUMBERS = 0
STATE_TYPE_IMAGE = 1

def msec_to_frames(milliseconds, fps=FPS):
   return fps * milliseconds / 1000.0

def frame_to_msec(frames, fps=FPS):
	return 1000.0*frames/fps

# hidden bird, to generate input for model
class H_Bird:
	WIDTH = HEIGHT = 32
	SINK_SPEED = 0.05
	CLIMB_SPEED = 0.2
	CLIMB_DURATION = 333.3
	
	def __init__(self, x, y, msec_to_climb, raw_images):
		self.x, self.y = x, y
		self.msec_to_climb = msec_to_climb
		self._img_wingup, self._img_wingdown = raw_images
		self._mask_wingup = self._img_wingup[:,:,3:4]/255
		self._mask_wingdown = self._img_wingdown[:,:,3:4]/255
		self._img_wingup = self._img_wingup[:,:,0:3]
		self._img_wingdown = self._img_wingdown[:,:,0:3]
		self.speed = H_Bird.SINK_SPEED
		
	def flap(self):
		self.msec_to_climb = H_Bird.CLIMB_DURATION
		
	def update(self, delta_frames=1):
		delta_time = frame_to_msec(delta_frames)
		if self.msec_to_climb>0:
			frac_climb_done = 1 -  self.msec_to_climb/H_Bird.CLIMB_DURATION # 0->1 => 0->pi => 1->-1 => 0->2
			self.speed = -H_Bird.CLIMB_SPEED * (1 - math.cos(frac_climb_done * math.pi))
			self.y +=  self.speed * delta_time
			self.msec_to_climb -= delta_time
		else: 
			self.speed = H_Bird.SINK_SPEED
			self.y += self.speed * delta_time
			
	def hit_edges(self):
		if self.y<0 or self.y + H_Bird.HEIGHT>WIN_HEIGHT:
			return True
		return False
			
	def get_image(self, clock_tick):
		if clock_tick%500>=250:
			return self._img_wingup
		else:
			return self._img_wingdown
			
	def get_mask(self, clock_tick):
		if clock_tick%500>=250:
			return self._mask_wingup
		else:
			return self._mask_wingdown
			image
			
	def draw(self, paper, clock_tick):
		start_x = int(self.x)
		end_x = start_x + H_Bird.WIDTH
		start_y = int(self.y)
		end_y = start_y + H_Bird.HEIGHT
		
		if start_y<0:
			mask = self.get_mask(clock_tick)[int(-start_y):H_Bird.HEIGHT]
			image = self.get_image(clock_tick)[int(-start_y):H_Bird.HEIGHT]
			end_y = int(end_y)
			start_y = 0
		elif end_y > WIN_HEIGHT:
			mask = self.get_mask(clock_tick)[0: int(WIN_HEIGHT-start_y)]
			image = self.get_image(clock_tick)[0: int(WIN_HEIGHT-start_y)]
			end_y = WIN_HEIGHT
			start_y = int(start_y)
		else:
			mask = self.get_mask(clock_tick)
			image = self.get_image(clock_tick)
			start_y = int(start_y)
			end_y = int(end_y)
		
		bird_img = image * mask + paper[start_y: end_y, start_x: end_x] * (1 - mask)
		paper[start_y: end_y, start_x: end_x] = bird_img
	
class H_PipePair:
	WIDTH = 80
	PIECE_HEIGHT = 32
	ADD_INTERVAL = 2500
	TOTAL_PIPE_BODY_PIECES = int(
		(WIN_HEIGHT -                  # fill window from top to bottom
		3 * H_Bird.HEIGHT -             # make room for bird to fit through
		3 * PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
		PIECE_HEIGHT)
	
	def __init__(self, pipe_end_img, pipe_body_img, bottom_pieces=np.random.randint(1, TOTAL_PIPE_BODY_PIECES)):
		self.x = float(WIN_WIDTH - 1)
		self.score_counted = False
		
		self.image = np.zeros([WIN_HEIGHT, H_PipePair.WIDTH, 4], dtype=np.float32)
		self.bottom_pieces = bottom_pieces
		self.top_pieces = H_PipePair.TOTAL_PIPE_BODY_PIECES - self.bottom_pieces
		
		start_x = 0
		end_x = H_PipePair.WIDTH
		
		# bottom pieces
		for i in range(1, self.bottom_pieces+1):
			start_y = WIN_HEIGHT - i * H_PipePair.PIECE_HEIGHT
			end_y = start_y + H_PipePair.PIECE_HEIGHT
			self.image[start_y: end_y, start_x: end_x] = pipe_body_img
		
		start_y = WIN_HEIGHT - self.bottom_pieces * H_PipePair.PIECE_HEIGHT - H_PipePair.PIECE_HEIGHT
		end_y = start_y + H_PipePair.PIECE_HEIGHT
		self.image[start_y: end_y, start_x: end_x] = pipe_end_img
		
		# top pipe
		for i in range(self.top_pieces):
			start_y = i * H_PipePair.PIECE_HEIGHT
			end_y = start_y + H_PipePair.PIECE_HEIGHT
			self.image[start_y:end_y, start_x:end_x] = pipe_body_img
		top_pipe_end_y = self.top_pieces * PipePair.PIECE_HEIGHT
		start_y = self.top_pieces * H_PipePair.PIECE_HEIGHT
		end_y = start_y + H_PipePair.PIECE_HEIGHT
		self.image[start_y: end_y, start_x: end_x] = pipe_end_img
		
		# compensate for added end pieces
		self.top_pieces += 1
		self.bottom_pieces += 1
		
		self.mask = self.image[:,:,3:4]/255
		self.image = self.image[:,:,0:3]
		
	def collides_with(self, bird):
		bird_upper_left_corner = [bird.x, bird.y]
		if self.x<bird_upper_left_corner[0]<self.x + H_PipePair.WIDTH and bird_upper_left_corner[1]<self.top_pieces*H_PipePair.PIECE_HEIGHT:
			return True
		
		bird_upper_right_corner = [bird.x + H_Bird.WIDTH, bird.y]
		if self.x<bird_upper_right_corner[0]<self.x + H_PipePair.WIDTH and bird_upper_right_corner[1] < self.top_pieces*H_PipePair.PIECE_HEIGHT:
			return True
			
		bird_lower_left_corner = [bird.x, bird.y + H_Bird.HEIGHT]
		if self.x<bird_lower_left_corner[0]<self.x + H_PipePair.WIDTH and bird_lower_left_corner[1]>WIN_HEIGHT - self.bottom_pieces*H_PipePair.PIECE_HEIGHT:
			return True
			
		bird_lower_right_corner = [bird.x + H_Bird.WIDTH, bird.y + H_Bird.HEIGHT]
		if self.x<bird_lower_right_corner[0]<self.x + H_PipePair.WIDTH and bird_lower_right_corner[1]>WIN_HEIGHT - self.bottom_pieces*H_PipePair.PIECE_HEIGHT:
			return True
		
		return False
	
	
	def is_visible(self):
		return -H_PipePair.WIDTH< self.x < WIN_WIDTH

	def update(self, delta_frames=1):
		self.x -=ANIMATION_SPEED * frame_to_msec(delta_frames)
		
	def draw(self, paper):
		start_x = self.x
		end_x = start_x + H_PipePair.WIDTH
		if start_x<0:
			mask = self.mask[:, int(-start_x): H_PipePair.WIDTH]
			image = self.image[:, int(-start_x): H_PipePair.WIDTH]
			end_x = int(end_x)
			start_x = 0
		elif end_x>WIN_WIDTH:
			mask = self.mask[:, 0: int(WIN_WIDTH-start_x)]
			image = self.image[:, 0: int(WIN_WIDTH-start_x)]
			end_x = WIN_WIDTH
			start_x = int(start_x)
		else:
			mask = self.mask
			image = self.image
			start_x = int(start_x)
			end_x = int(end_x)
		pipe_img = image * mask + paper[:, start_x: end_x] * (1 - mask)
		paper[:, start_x: end_x] = pipe_img

# create environment without clock ticking
class QuickEnvironment:

	SCREENSHOT_INTERVAL = 100
	MAX_DURATION = 100000 # 100 secs
	
	def __init__(self, image_folder):
		self.images = self.load_raw_images(image_folder=image_folder)
		self.bird = H_Bird(50, int((WIN_HEIGHT - H_Bird.HEIGHT)/2), 2, [self.images['bird-wingup'], self.images['bird-wingdown']])
		self.clock_tick = 0
		self.frame_count = 0
		self.pipes = []
		self.background = np.concatenate([self.images['background'], self.images['background']], axis=1)
		self.background = self.background[:,:,0:3]

	def load_raw_images(self, image_folder='./cimages/'):
		def load_raw_image(img_file_name):
			img = np.float32(cv2.imread(image_folder + img_file_name, cv2.IMREAD_UNCHANGED))
			if img.shape[2]==3:
				img = np.concatenate([img, np.ones([img.shape[0], img.shape[1], 1], dtype=np.float32)*255], axis=2)
			return img

		return {
			'background':  load_raw_image('background.png'),
			'pipe-end': load_raw_image('pipe_end.png'), 
			'pipe-body': load_raw_image('pipe_body.png'), 
			'bird-wingup': load_raw_image('bird_wing_up.png'), 
			'bird-wingdown': load_raw_image('bird_wing_down.png')}	

	def reset(self, state_type=STATE_TYPE_IMAGE):
		self.clock_tick = 0
		self.frame_count = 0
		self.bird.y = np.random.uniform(low=WIN_HEIGHT/4, high=3*WIN_HEIGHT/4-H_Bird.HEIGHT)
		self.bird.msec_to_climb = 2
		self.pipes.clear()
		return self.get_state(state_type)
		
	def screenshot(self):
		paper = self.background.copy()
		self.bird.draw(paper, self.clock_tick)
		for pipe in self.pipes:
			pipe.draw(paper)
		return paper
		
	def get_duration(self):
		return frame_to_msec(self.frame_count)/1000

	def get_next_pipe_pos(self):
		pipe_pos = WIN_WIDTH
		for pipe in self.pipes:
			if pipe.x + H_PipePair.WIDTH > self.bird.x:
				pipe_pos = pipe.x
				break
		return pipe_pos

	def update(self, action=ACTION_DO_NOTHING, state_type=STATE_TYPE_IMAGE):
		# increase clock tick
		self.clock_tick+=16.667 # 60 fps => clock tick += 1000/60
		t_action = action
		# init reward, done
		reward = REWARD_SURVIVE
		done = False
		
		n_frames = int(np.round(msec_to_frames(QuickEnvironment.SCREENSHOT_INTERVAL)))
		for i in range(n_frames):
			# after every 3000 ms add a piple
			if self.frame_count%msec_to_frames(PipePair.ADD_INTERVAL)==0:
				self.pipes.append(H_PipePair(self.images['pipe-end'], self.images['pipe-body'], np.random.randint(1, H_PipePair.TOTAL_PIPE_BODY_PIECES)))
			
			# update bird	
			if t_action==ACTION_FLAP:
				self.bird.flap()
				t_action = ACTION_DO_NOTHING
			self.bird.update()
			

			# check bird state
			if self.bird.hit_edges():
				reward = REWARD_DEAD
				done = True
				return None, reward, done
				
			# update pipe
			for pipe in self.pipes:
				pipe.update()
				if pipe.collides_with(self.bird):
					reward = REWARD_DEAD
					done = True
					return None, reward, done
					
			# check if episode is too long
			if self.frame_count>msec_to_frames(QuickEnvironment.MAX_DURATION):
				reward = REWARD_SURVIVE
				done = True
				return None, reward, done
			
			if not self.pipes[0].is_visible():
				self.pipes.pop(0)
					
			# increase frame count
			self.frame_count += 1
		return self.get_state(state_type), reward, done
	
	def get_state(self, state_type=STATE_TYPE_IMAGE):
		if state_type==STATE_TYPE_NUMBERS:
			return [self.bird.y/191 - 1, self.bird.speed/2/H_Bird.CLIMB_SPEED, self.get_next_pipe_pos()/223 - 1]
		else: # this state type does not contain speed information
			return self.screenshot()/255
		
class Bird(pygame.sprite.Sprite):

	WIDTH = H_Bird.WIDTH
	HEIGHT = H_Bird.HEIGHT
	SINK_SPEED = H_Bird.SINK_SPEED
	CLIMB_SPEED = H_Bird.CLIMB_SPEED
	CLIMB_DURATION = H_Bird.CLIMB_DURATION
	
	def __init__(self, x, y, msec_to_climb, images):
		super(Bird, self).__init__()
		self.x, self.y = x, y
		self.msec_to_climb = msec_to_climb
		self._img_wingup, self._img_wingdown = images
		self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
		self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)
		self.speed = Bird.SINK_SPEED
	
	def update(self, delta_frames=1):
		delta_time = frame_to_msec(delta_frames)
		if self.msec_to_climb>0:
			frac_climb_done = 1 -  self.msec_to_climb/Bird.CLIMB_DURATION
			self.speed = -Bird.CLIMB_SPEED * (1 - math.cos(frac_climb_done * math.pi))
			self.y += self.speed * delta_time
			self.msec_to_climb -= delta_time
		else: 
			self.speed = Bird.SINK_SPEED
			self.y += self.speed * delta_time
			
	def flap(self):
		self.msec_to_climb = Bird.CLIMB_DURATION-1
			
	@property
	def image(self):
		if pygame.time.get_ticks()%500>=250:
			return self._img_wingup
		else:
			return self._img_wingdown
	
	@property		
	def mask(self):
		if pygame.time.get_ticks()%500>=250:
			return self._mask_wingup
		else:
			return self._mask_wingdown
	
	@property
	def rect(self):
		return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)
	
class PipePair(pygame.sprite.Sprite):
	WIDTH = H_PipePair.WIDTH
	PIECE_HEIGHT = H_PipePair.PIECE_HEIGHT
	ADD_INTERVAL = H_PipePair.ADD_INTERVAL
	TOTAL_PIPE_BODY_PIECES = H_PipePair.TOTAL_PIPE_BODY_PIECES
	def __init__(self, pipe_end_img, pipe_body_img, bottom_pieces=np.random.randint(0, H_PipePair.TOTAL_PIPE_BODY_PIECES)):
		self.x = float(WIN_WIDTH - 1)
		self.score_counted = False
		
		self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
		self.image.convert()
		self.image.fill((0,0,0,0))
		
		total_pipe_body_pieces = int(
			(WIN_HEIGHT -                  # fill window from top to bottom
			 3 * Bird.HEIGHT -             # make room for bird to fit through
			 3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
			PipePair.PIECE_HEIGHT          # to get number of pipe pieces
		)
		self.bottom_pieces = bottom_pieces
		self.top_pieces = total_pipe_body_pieces - self.bottom_pieces
		# bottom pieces
		for i in range(1, self.bottom_pieces+1):
			piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
			self.image.blit(pipe_body_img, piece_pos)
		bottom_pipe_end_y = WIN_HEIGHT - self.bottom_pieces*PipePair.PIECE_HEIGHT
		bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
		self.image.blit(pipe_end_img, bottom_end_piece_pos)
		
		# top pipe
		for i in range(self.top_pieces):
			self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
		top_pipe_end_y = self.top_pieces * PipePair.PIECE_HEIGHT
		self.image.blit(pipe_end_img, (0, top_pipe_end_y))
		
		# compensate for added end pieces
		self.top_pieces += 1
		self.bottom_pieces += 1

		# for collision detection
		self.mask = pygame.mask.from_surface(self.image)
		
	def is_visible(self):
		return -PipePair.WIDTH< self.x < WIN_WIDTH
		
	@property	
	def rect(self):
		return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)
		
	def update(self, delta_frames=1):
		self.x -=ANIMATION_SPEED * frame_to_msec(delta_frames)
		
	def collides_with(self, bird):
		bird_upper_left_corner = [bird.x, bird.y]
		
		if self.x<bird_upper_left_corner[0]<self.x + PipePair.WIDTH and bird_upper_left_corner[1]<self.top_pieces*PipePair.PIECE_HEIGHT:
			return True
		
		bird_upper_right_corner = [bird.x + Bird.WIDTH, bird.y]
		if self.x<bird_upper_right_corner[0]<self.x + PipePair.WIDTH and bird_upper_right_corner[1] < self.top_pieces*PipePair.PIECE_HEIGHT:
			return True
			
		bird_lower_left_corner = [bird.x, bird.y + Bird.HEIGHT]
		if self.x<bird_lower_left_corner[0]<self.x + PipePair.WIDTH and bird_lower_left_corner[1]>WIN_HEIGHT - self.bottom_pieces*PipePair.PIECE_HEIGHT:
			return True
			
		bird_lower_right_corner = [bird.x + Bird.WIDTH, bird.y + Bird.HEIGHT]
		if self.x<bird_lower_right_corner[0]<self.x + PipePair.WIDTH and bird_lower_right_corner[1]>WIN_HEIGHT - self.bottom_pieces*PipePair.PIECE_HEIGHT:
			return True
		
		return False

class Environment:

	SCREENSHOT_INTERVAL = QuickEnvironment.SCREENSHOT_INTERVAL
	
	def __init__(self, image_folder='./images/'):
		pygame.init()
		self.display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
		pygame.display.set_caption('Reinforcement Environment')
		self.images = self.load_images(image_folder=image_folder)
		self.clock = pygame.time.Clock()
		self.bird = Bird(
			50, 
			int((WIN_HEIGHT - Bird.HEIGHT)/2),
			2, 
			[self.images['bird-wingup'], self.images['bird-wingdown']])
		self.pipes = []
		
	def load_images(self, image_folder='./images/'):
		def load_image(img_file_name):
			img = pygame.image.load(image_folder + img_file_name)
			img.convert()
			return img
				
		return {
			'background':  load_image('background.png'),
			'pipe-end': load_image('pipe_end.png'), 
			'pipe-body': load_image('pipe_body.png'), 
			'bird-wingup': load_image('bird_wing_up.png'), 
			'bird-wingdown': load_image('bird_wing_down.png')}
			
	def should_screenshot(self):
		if self.frame_count%msec_to_frames(Environment.SCREENSHOT_INTERVAL)==0:
			return True
		return False
	
	def screenshot(self):
		img = pygame.surfarray.array3d(self.display_surface)
		img = img.swapaxes(0,1)
		return img
		
	def reset(self):
		self.clock_tick = 0
		self.frame_count = 0
		self.bird.y = int((WIN_HEIGHT - H_Bird.HEIGHT)/2)
		self.bird.msec_to_climb = 2
		self.pipes.clear()

	def run(self):
		done = False
		paused = False
		frame_clock = 0
		while not done:
			self.clock.tick(FPS)
			clock_tick = pygame.time.get_ticks()
			
			# add pipe after every 3 secs
			if not (paused or frame_clock%msec_to_frames(PipePair.ADD_INTERVAL)):
				self.pipes.append(PipePair(self.images['pipe-end'], self.images['pipe-body'], np.random.randint(1, PipePair.TOTAL_PIPE_BODY_PIECES)))
			
			# handle events
			for e in pygame.event.get():
				if e.type==KEYUP:
					if e.key==K_ESCAPE:
						done=True
						break
					elif e.key==K_UP:
						self.bird.flap()
					elif e.key==K_p:
						paused = not paused
				elif e.type==QUIT:
					done=True
					break
					
			# do not do anything if paused 
			if paused:
				continue
				
			if self.bird.y<=0 or self.bird.y>=WIN_HEIGHT-Bird.HEIGHT:
				done = True
				
			for x in [0, WIN_WIDTH/2]:
				self.display_surface.blit(self.images['background'], (x, 0))
			
			
			if len(self.pipes)>0 and not self.pipes[0].is_visible():
				self.pipes.pop(0)
				
			for pipe in self.pipes:
				pipe.update()
				self.display_surface.blit(pipe.image, pipe.rect)
				if pipe.collides_with(self.bird):
					done = True
					break
			
			self.bird.update()
			self.display_surface.blit(self.bird.image, self.bird.rect)
			
			# screenshot
			'''
			if not frame_clock%msec_to_frames(Environment.SCREENSHOT_INTERVAL):
				cv2.imwrite('test{:06d}.jpg'.format(frame_clock), self.screenshot())
			'''
			
			pygame.display.update()
			frame_clock+=1
			
		pygame.quit()

'''
env = Environment('./images/')
env.run()

env = QuickEnvironment('./images/')
env.reset()
done = False
count = 0
while not done:
	reward, done = env.update(action=np.random.randint(2))
	img = env.screenshot()
	cv2.imwrite('test{:06d}.jpg'.format(count), img)
	count += 1
	if count>1000:
		break
'''