import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        new_pixels = []
        trans_matrix = np.array([[1, 0, 0, 0], 
                                [0, 0, 1, -1.5],
                                [0, -1, 0, 0], 
                                [0, 0, 0, 1]])
        intrinsic_matrix = np.array([[256, 0, 256], 
                            [0, 256, 256], 
                            [0, 0, 1]])
        homo_intrinsic_matrix = np.array([[256, 0, 256, 0], 
                            [0, 256, 256, 0], 
                            [0, 0, 1, 0]])
        print(points)
        for i in range(4):
            current_point = [points[i][0], points[i][1], 1]
            #image cor to camera cor
            c2_camera_point = 2.5 * np. linalg. inv(intrinsic_matrix) @ current_point
            print(c2_camera_point)
            c2_camera_point = np.append(c2_camera_point, [1])

            #change c2 cor to c1 cor
            c1_camera_point = trans_matrix@c2_camera_point
            #camera cor to image cor
            new_point = homo_intrinsic_matrix @ c1_camera_point
            u = int(new_point[0]/new_point[2])
            v = int(new_point[1]/ new_point[2])
            new_pixels.append([u, v])
        print(new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
