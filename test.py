from utils import *

landscape = load_image("family.jpeg")
landscape_resized = resize_with_borders(landscape, 512)
print(landscape_resized.shape)
show_image(landscape_resized)

portrait = load_image("portrait.jpg")
portrait_resized = resize_with_borders(portrait, 512)
print(portrait_resized.shape)
show_image(portrait_resized)
