import os
import numpy as np
import pyvista as pv
import shapeworks as sw
import scipy
import matplotlib.pyplot as plt
import glob

'''
get_image helper
'''
def blur(img, size):
	blur = scipy.ndimage.filters.gaussian_filter(img, size)
	return blur

'''
get_image helper
'''
def apply_noise(img, foreground_mean, foreground_var, background_mean, background_var):
	background_indices = np.where(img < 0.5)
	foreground_indices = np.where(img > 0.5)
	img = img*(foreground_mean-background_mean)
	img = img + np.ones(img.shape)*background_mean
	foreground_noise = np.random.normal(0, foreground_var**0.5, img.shape)
	foreground_noise[background_indices] = 0
	background_noise = np.random.normal(0, background_var**0.5, img.shape)
	background_noise[foreground_indices] = 0
	noisy_img = img + foreground_noise + background_noise
	return noisy_img




'''
Generates image by blurring and adding noise to segmentation
'''
def generate_images(generated_directories, blur_factor_max, foreground_mean, foreground_var, background_mean, background_var):

	if type(generated_directories) != list:
		generated_directories = [generated_directories]

	all_ims = []

	# Get all images from all directories
	for i in range(len(generated_directories)):
		segs = glob.glob(generated_directories[i]+"/segmentations/*.nrrd")

		imgDir = generated_directories[i] + 'images/'
		imgDir2D = generated_directories[i] + 'images2D/'
		os.makedirs(imgDir, exist_ok=True)
		os.makedirs(imgDir2D, exist_ok=True)

		index = 1
		for seg in segs:

			blur_factor = np.random.randint(1,blur_factor_max+1)

			print("Generating image " + str(index) + " out of " + str(len(segs)))
			name = seg.replace('segmentations/','images/').replace('.nrrd', '_blur' + str(blur_factor) + '.nrrd')
			name2D = seg.replace('segmentations/','images2D/').replace('.nrrd', '_blur' + str(blur_factor) + '.png')

			img = sw.Image(seg)
			origin = img.origin()
			img_array = blur(img.toArray(), blur_factor)
			img_array = apply_noise(img_array, foreground_mean, foreground_var, background_mean, background_var)
			img_array = np.float32(img_array)

			z_size, _, _ = img_array.shape
			middle_slice = z_size//2
			img2D = img_array[middle_slice, :, :]
			img2D = np.flipud(img2D)
			img2D = img2D.astype(np.uint8)
			
			plt.imsave(name2D, img2D, cmap="gray")
			# import pdb;pdb.set_trace()

			img = sw.Image(np.float32(img_array)).setOrigin(origin)
			img.write(name,compressed=True)
			index += 1
		

	return all_ims




def generate(out_dir='../datasets/torus_bump/', num_samples=300):

	mesh_dir = out_dir+"/meshes/"
	seg_dir = out_dir+"/segmentations/"

	os.makedirs(mesh_dir, exist_ok=True)
	os.makedirs(seg_dir, exist_ok=True)

	# Create torus
	tor = pv.ParametricTorus().triangulate()
	box = pv.Box(level=2).scale([4,2,4], inplace=False).translate((0, -2, 0), inplace=True).triangulate()
	# Add bump
	ellipsoid = pv.ParametricEllipsoid(1,1,1).translate((0, 3.5, 0), inplace=True).scale((0.4, 0.4, 0.4), inplace=True).triangulate()
	ellipsoid.flip_normals()
	tor = tor.boolean_union(ellipsoid)

	for i in range(num_samples):
		# Rotate by random angle
		angle = np.random.randint(-90, 90)
		rot = tor.rotate_z(angle, inplace=False)
		# Remove half using box
		torus = rot.boolean_difference(box).scale([150,150,150], inplace=True)
		try:
			torus.save(mesh_dir + 'id' + str(i).zfill(4) + '_torus.vtk')
			

			print(f'Saving samples mesh: {i}')

			mesh = sw.Mesh(mesh_dir + 'id' + str(i).zfill(4) + '_torus.vtk')
			bb = mesh.boundingBox()
			bb.min -= 50
			bb.max += 50
			mesh.toImage(region=bb, spacing =[1,1,1]).write(seg_dir + 'id' + str(i).zfill(4) + '_torus.nrrd')
		except:
			print("Can't save")

		



def particles(in_dir="torus_bump_remesh/", out_dir="torus_bump_particles/"):
	for file in sorted(os.listdir(in_dir)):
		points = sw.Mesh(in_dir+file).points()
		np.savetxt(out_dir+file.replace(".vtk",".particles"), points)

def remesh(in_dir="torus_bump/", out_dir="torus_bump_remesh/"):
	for file in sorted(os.listdir(in_dir)):
		mesh = sw.Mesh(in_dir+file).remesh(numVertices=1200, adaptivity=0.0).write(out_dir+file)
		print(mesh.points().shape[0])





if __name__ == '__main__':

	out_dir = '../datasets/torus_bump_300/'
	generate_images(generated_directories = out_dir, blur_factor_max=5, foreground_mean=180, foreground_var=30, background_mean=80, background_var=30)
