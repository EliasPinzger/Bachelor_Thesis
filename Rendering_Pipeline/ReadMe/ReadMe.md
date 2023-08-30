# **Generation Of Synthetic Datasets**

This project is designed to create synthetic datasets with traces. The purpose of these sets is to train CNNs on them
and check which of the traces are used. This tutorial will focus solely on the rendering pipeline and not the usage of
the sets. It explains how the pipeline is structured and the possibilities to modify and extend it.

- [1. Introduction](#1-introduction)
- [2. Blender](#2-blender)
    * [2.1 Scripting](#21-scripting)
- [3. Traces](#3-traces)
    * [3.1 Shape](#31-shape)
    * [3.2 Texture](#32-texture)
    * [3.3 Color](#33-color)
    * [3.4 Scale](#34-scale)
    * [3.5 Background](#35-background)
    * [3.6 Lighting](#36-lighting)
- [4. Camera](#4-camera)
- [5. Data Generation](#5-data-generation)
- [6. Expand Attributes](#6-expand-attributes)
    * [6.1 Lighting, Scales, Camera Perspectives and Lights](#61-lighting-scales-camera-perspectives-and-lights)
    * [6.2 Shapes and Background Planes](#62-shapes-and-background-planes)
    * [6.3 Backgrounds and Textures](#63-backgrounds-and-textures)
- [7.Rendering](#7-rendering)
    * [7.1 Rendering Options](#71-rendering-options)
    * [7.2 Labeling](#72-labeling)
- [8. Program Execution](#8-program-execution)
    * [8.1 Terminal](#81-terminal)
    * [8.2 Blender GUI](#82-blender-gui)
- [9. Device](#9-device)
    * [9.1 Script](#91-script)
    * [9.2 Blender GUI](#92-blender-gui)

<!-- toc -->

### **1. Introduction**

This project is part of the bachelor thesis "What Traces Does My Image Recognition CNN Learn?". It uses Blender to
create synthetic datasets for object recognition. If desired, traces can be set which a machine learning model can use
for classification. The following only deals with the dataset type *geometric shapes* because the procedure for
*planets* is largely identical. For this tutorial, it is expected that the file geometric.blend is opened in Blender.

Used programs:

Blender 3.5.0

Python 3.10

### **2. Blender**

This project uses the rendering toolkit Blender to render the images. In Blender, a scene is set up which contains
meshes
of the
different shapes, light sources, background planes and a camera coupled to an axis.
What that is and how to adjust the given scene is explained in later sections. Afterward, this scene can be used to
render the images for a dataset. The main part of the program is to change the scene after an image was rendered
to generate different images in each step.

Blender has multiple editors to display or modify data. During this tutorial, some of those editors are needed.
To access them, click on the first button in the upper left corner of a header.
This will open a list of all editors, as can be seen in the image below.

<img src="images/editors.png" alt="drawing" width="800"/>

#### **2.1 Scripting**

This project is designed around the scripting feature of Blender. With this, it is possible to use python code to change
the scene as needed.
To access Blenders internal data, the Python module bpy is needed.
To execute the scripts, the embedded Python interpreter of Blender can be used. Blender also has a basic text editor,
however, it is recommended to use an IDE.

### **2.2 Scene Structure**

The scene in Blender is one of the key elements and a good starting point to get an overview of the
project.
The scene must have a certain structure so that everything can be easily accessed in the python script.
This structure can be seen in the Outliner. The lights must be in a collection "Lights". The shapes in a collection
"Shapes" and the background planes in "Background Planes." The axis which is
coupled with, the camera is on the same level as the previously mentioned collections.
It is important that the name of the collections is the same as mentioned before and that the axis is named "Axis". An
example of such a structure can be seen in the next image.

<img src="images/outline.png" alt="drawing" width="800"/>

### **3. Traces**

Traces are properties which can be used for classification.
As mentioned in the [Introduction](#1-introduction) the aim of the project is to generate datasets with various traces.
In the following, all supported attributes which can be used as traces will shortly be explained.
To get a better understanding, either some knowledge of Blender
is needed or the section [Expand Attributes](#6-expand-attributes) can be read. It should be noted that the order of the
traces has no effect because the program orders them indirectly.

#### **3.1 Shape**

The different shapes are meshes which are extracted directly from the scene. All shapes are at the same location.

#### **3.2 Texture**

The textures are contained in the Blender materials, and so they are obtained from the blend file.
All materials which contain the String 'Texture' in their name are used as textures for the shapes.
Such a material must contain a Color Ramp Node. What this is will be explained
in [Expand Attributes](#6-expand-attributes).

#### **3.3 Color**

A color is a tuple of the form (Red, Green, Blue, Alpha) where each
of these values should be between 0 and 1. The colors are always bound to a texture.
More exactly, a color is assigned to a texture through a specific Color Ramp Node with the name 'Color'
which is used to change the color of the texture.

#### **3.4 Scale**

The attribute Scale determines the size of an object. Usually a shape has a scale of (X=1, Y=1, Z=1). When changing the
scale each dimension is equally adjusted. Additionally, the shape will move on the z-axis. Otherwise,
the shape would pierce through the background plane.

#### **3.5 Background**

The background is nearly identical to the trace texture and is therefore extracted from the blend file.
The difference is that a background does contain the String 'background' in its name and does not need a Color
Ramp Node. Such a background is assigned to the background planes. In this project, only one background plane is used
which is directly under the shapes.

#### **3.6 Lighting**

The Trace Lightning changes the power of each light source in the collection Lights. In doing so
the same power is assigned to each light source.

### **4. Camera**

In contrast to the possible traces, the position of the camera cannot be used for
classification. The camera always rotates around the
center of the scene. This happens indirectly because the camera is attached to an
axis which is rotated instead. This axis is located at the origin of the scene.
The camera can observe the scene from different angles and heights. For this, the angles beta and gamma are used. Beta
is used to rotate around the x-axis and gamma for the z-axis.
However, in this project there is a background directly below the shapes so the camera shouldn't
turn below a shape.

### **5. Data Generation**

In the data generation process, each possible combination of attributes is generated. If needed the number of values per
attribute can be
expanded. How this works is explained in the section [Expand Attributes](#6-expand-attributes). Should one
of the attributes be used as a trace (camera perspective cannot be used as a trace) then the values of this attribute
are distributed so that each classification object gets one distinct value. Thus, the dataset shrinks or
stays the same size with every added trace. It should be noted that there must be enough values of a trace for each
class. Should there be more values than classes, only the first values are used. At least one trace should be selected
otherwise, all classes will have the same images. And even if this would not be the case, the machine learning model
would perform poorly if it does not find an unexpected trace.


### **6. Expand Attributes**

This section explains how the values of the attributes can be expanded.

#### **6.1 Lighting, Scales, Camera Perspectives and Lights**

The values for Lighting, Scale and Camera Perspective only consist of a list
of numbers.
To extend them only the list must be adjusted, for this project these values are randomly generated and loaded from a
file.

To add further lights to a scene, click Add->Light in the 3D Viewport and choose one of the available types.
Afterward, click on the light and tap G to move it around or use the coordinates at Properties->Object Properties.

#### **6.2 Shapes and Background Planes**

Shapes are meshes which can be added by either creating them in
Blender or by importing them into Blender from another program or a library.
This is a topic in itself, to get more information a specific tutorial should be considered.
Like Shapes the background planes are meshes, it is not necessary to use planes, however for this project
planes are the most suitable meshes. To add such a plane, click Add->Mesh->Plane in the 3D Viewport. It can be moved
like
a light
which was explained in the [previous section](#61-lighting-scales-camera-perspectives-and-lights).

#### **6.3 Backgrounds and Textures**

As already mentioned in [Texture](#32-texture) and [Background](#35-background).
Textures and backgrounds are both materials. Whereby they are
distinguished by their name and that textures have a Color Ramp Node with which the color
can be changed. To create a material
a mesh must be selected, then click Properties->Material Properties and use the button New (if the mesh has already a
material
assigned
then unlink it with the cross button). In the Base Color tab, select Image Texture and assign a texture.

<img src="images/image_texture.png" alt="drawing" width="800"/>

Another possibility to add a material is to download one from a library like BlenderKit. It should be noted that if a
material has no user (a user is for example a mesh which uses this material) it is not loaded by the program and will
be automatically
deleted. To avoid this, a Fake User can be added to the material. In the Outliner editor, select at the second tab
Blender File. There in Materials, right-click on the material and click Add Fake User. Alternatively, click the empty
shield in Properties->Material Properties for the current material. There at 'Browse Material to be linked' all
materials with a Fake User have an 'F' in front of their name.

<img src="images/blender_file.png" alt="drawing" width="800"/>

Keep in mind that the materials are by default not displayed in the viewport.
To see them select Viewport Shading in the top right corner of the 3D Viewport.

Once the material has been created, it can be used as a background by naming it so that it contains the String
'Background'.
To use it as a texture, a Color Ramp Node must be added.
This can be done in the Shader Editor with Add->Converter->ColorRamp.
This node must then be placed between the Image Texture Node and the Principled BSDF. If it is a downloaded material
the placement can vary greatly. Linking these nodes should happen automatically. Afterward, this node must be named
'Color'.
How this should look like can be seen in the next image.

<img src="images/color_ramp_node.png" alt="drawing" width="800"/>

### **7. Rendering**

Since there is always only one shape in an image
all of them are at the same location, and all except one are set to be hidden during rendering. All other scene objects
are never hidden. Furthermore, the program does not disable the hiding of objects. That means if one of them is hidden
in Blender than it
will not be visible in the generated images. The objects can be hidden for rendering in the Outliner with the camera
symbol next to each object.

#### **7.1 Rendering Options**

Some render settings can be changed in the Render class.
Like the resolution of the rendered images and the percentage scale
for the render resolution. In addition, the samples can be adjusted, the higher
this number the more accurate the light calculation, but it takes
longer to render an image. It should be noted that in this project the
Rendering Engine Cycles is used. Blender has several rendering engines
to choose from, but this out of scope for this tutorial.

#### **7.2 Labeling**

The images are labeled by saving them in folders with the same name as the class they belong to.

### **8. Program Execution**

There are several options to run the project or generally run python scripts with Blender.
Before that, however, a few things need to be adjusted. First, the global variable source_path has to be set,
this variable contains the path where the source files are located. This is necessary because some python files are
imported dynamically.
In addition, the filepath must be set in the main function when creating a render instance, there the dataset will be
created. In this project, the values of the continuous attributes are randomly generated. To generate them, just execute
the file random_attributes_values.py
with the appropriate filepath and the wanted values. Also adjust the filepath for the generated file in
main.py.

As a side note, the generation of a dataset can take a long time. Should it happen that
the program is interrupted because the system crashes or the computer is needed for another purpose than
the rendering process can be continued later on. For this, the script must be run again with the same settings.
If the folder structure is found, it can be selected whether the existing data should be used. It is important
that if any changes have been made, it cannot be guaranteed that the generated dataset is correct. In addition, changes
in the destination folder should be made with care because to find an
entry point only the number of files contained in the folder is used. If the program is interrupted, it is possible that
the last rendered image is
corrupted. This
happens quite seldom, but it is recommended to delete the last image manually.

Once this initial setup is done, one of the following two approaches can be used (How to use a GPU for execution
is explained in the section [Device](#9-device).

#### **8.1 Terminal**

The program can be started via a terminal. This can be done either by changing to the directory where
Blender was installed or from anywhere if Blender was added to the system PATH. The program can then
be executed with this command: blender `<path of blend file>` --background --python `<path of main.py>`
Example with rendering pipeline as current directory, Blender added to system PATH and usage of relative paths:

blender \resources\geometric.blend --background --python \src_g\main.py

--background will start blender without GUI, which is recommended because of performance reasons.

#### **8.2 Blender GUI**

The program can be run using the Blender GUI. First the Blender File has to be opened and then the integrated Text
Editor opened. There, either the main.py can be opened as a text data block or a
new text data block can be created where the contents of main.py are copied to. To interact with the program
the system console is needed. It can be activated at Window->Toggle System Console (see image below), this must be done
before starting
the program. If this has not been done, the output cannot be viewed and because user input is needed, the program hangs.
In addition, it cannot be opened later on because Blender is unresponsive while a script is running. To run the script
click
Run Script.

<img src="images/system_console.png" alt="drawing" width="800"/>

### **9. Device**

The rendering of datasets is a computation intensive process. However, with the usage of a GPU the processing
time can be significantly reduced. In the following, two approaches are presented with which a GPU can be used.

#### **9.1 Script**

Before the render method is called, the method enable_gpus with one of the following device types ('CUDA', 'OPTIX',
'HIP', 'ONEAPI') must be invoked. Which device type is the correct one depends
on the GPU and can be looked up
[here](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html).
Alternatively, it is possible to look it up in Blender, where this can be found is
explained in the second approach. Do not use this method if the system has no GPU or the device type is incorrect
because this method also sets the tile size. If the usage of a GPU did not work, the CPU is used instead and the
inappropriate tile size will slow down the rendering.

#### **9.2 Blender GUI**

The easiest way to determine which CPU or GPU to use is within the Blender GUI.
The procedure for this is to click Edit->Preferences->System and there at
Cycles Render Devices all devices are listed under their corresponding type.
There the desired device must be selected and then go to Properties->Render Properties. At device
select whether a CPU or GPU should be used. At Performance->
Memory tiling can be activated and a tile size be set (Recommended: (CPU: 64, GPU: 256)).

<img src="images/devices.png" alt="drawing" width="800" height="400"/>
<img src="images/gpu.png" alt="drawing" width="800" height="400"/>
