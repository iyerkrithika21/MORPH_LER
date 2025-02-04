import os
import numpy as np
import pyvista as pv
import shapeworks as sw
def generate_ellipsoid(out_dir='ellipsoid_bump/', num_samples=30):
	# Create ellipsoid
	points = 512
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for i in range(num_samples):
		rx = np.random.randint(low =15,high=25,size =1)[0]
		ry = np.random.randint(low =5,high=15,size =1)[0]
		rz = np.random.randint(low =5,high=15,size =1)[0]

		
		
		ellipsoid = pv.ParametricEllipsoid(rx,ry,rz,center=[0,0,0], ).triangulate() #u_res=8, v_res=8, w_res=8
		# ellipsoid.flip_normals()

		
		
		
		ellipsoid.save(out_dir + 'id' + str(i).zfill(4) + '_ellipsoid.vtk')
def generate(out_dir='ellipsoid_bump/', num_samples=15):
	# Create ellipsoid
	points = 512
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for i in range(num_samples):
		rx = np.random.randint(low =15,high=25,size =1)[0]
		ry = np.random.randint(low =5,high=15,size =1)[0]
		rz = np.random.randint(low =5,high=15,size =1)[0]

		
		# Set of all spherical angles:
		u = np.linspace(0, 2 * np.pi, points)
		v = np.linspace(0, np.pi, points)

		# Cartesian coordinates that correspond to the spherical angles:
		# (this is the equation of an ellipsoid):
		x = rx * np.outer(np.cos(u), np.sin(v))
		y = ry * np.outer(np.sin(u), np.sin(v))
		z = rz * np.outer(np.ones_like(u), np.cos(v))
		x = x.flatten()
		y = y.flatten()
		z = z.flatten()
		particles = np.column_stack((x,y,z))
		ellipsoid = pv.ParametricEllipsoid(rx,ry,rz,center=[0,0,0], ) #u_res=8, v_res=8, w_res=8
		ellipsoid.flip_normals()

		# Add bump
		sample_idx = np.random.randint(points,size=1)[0]
		center = particles[sample_idx,:]
		radius = 4
		sphere = pv.Sphere(radius,center=center, )
		# sphere = pv.ParametricEllipsoid(radius,radius,radius,center=center, ).triangulate()   #u_res=8, v_res=8, w_res=8
		ellipsoid_bump = ellipsoid.boolean_union(sphere)
		ellipsoid_bump = ellipsoid_bump.triangulate()
		
		
		ellipsoid_bump.save(out_dir + 'id' + str(i).zfill(4) + '_ellipsoid_bump.vtk')

		# ellipsoid.plot(color='w', smooth_shading=True)
		# pl = pv.Plotter()
		# _ = pl.add_mesh(ellipsoid_bump, color='tan', style='wireframe', line_width=3)
		# # _ = pl.add_mesh(ellipsoid, color='blue', style='wireframe', line_width=3)
		# # _ = pl.add_mesh(sphere, color='red', style='wireframe', line_width=3)
		# pl.show_axes()
		# _ = pl.show_grid()
		# pl.show()

def particles(in_dir="ellipsoid_bump_remesh/", out_dir="ellipsoid_bump_particles/"):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for file in sorted(os.listdir(in_dir)):
		points = sw.Mesh(in_dir+file).points()
		idx = np.random.randint(len(points), size=1024)
		points = points[idx, :]
		np.savetxt(out_dir+file.replace(".vtk",".particles"), points)

def remesh(in_dir="ellipsoid_bump/", out_dir="ellipsoid_bump_remesh/"):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for file in sorted(os.listdir(in_dir)):
		print(in_dir+file)
		mesh = sw.Mesh(in_dir+file).remesh(numVertices=1024, adaptivity=1.0).write(out_dir+file)
		print(mesh.points().shape[0])

# generate_ellipsoid()
generate()
remesh()
particles()