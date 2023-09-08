import os

import imageio

#path_to_images = os.path.join("vis", "gif_frames_2k_epochs", "IMG_0753")
#gif_name = "test_400_epochs"


def make_gif(path_to_images, gif_name):
    file_names = [file_name for file_name in os.listdir(path_to_images)]
    print(file_names)
    file_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    print("\nGenerating GIF from {} images stored at {}\n".format(len(file_names), path_to_images))

    images = []
    for file_name in file_names:
        print("\t Frame: {}".format(file_name))
        images += [imageio.imread(os.path.join(path_to_images, file_name))]

    print("\n Generating GIF at {}... \n".format(path_to_images))
    imageio.mimsave(os.path.join(path_to_images, "{}.gif".format(gif_name)), images)
    print(" GIF generated successfully! \n")


# if __name__ == '__main__':
#    main()
