import cv2
import argparse
import os
from pathlib import Path
import process_sentinel2
from functools import partial
import numpy as np
import json

COLORS = {
    'valid water': (0,255,255)
}
#PATH_OUTPUT = "area_definition.json"

def add_point(points, width, height, coords, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        new_point = {
            'x': x / width,
            'y': y / height
        }

        new_coord = {
            'x': x,
            'y': y
        }
        print(new_coord)
        points.append(new_point)
        coords.append(new_coord)

def draw_areas(img, defined_areas):
    height, width , _ = img.shape
    for defined_area_name, defined_area_points in defined_areas.items():
        pts = np.array([[p['x']*width, p['y']*height] for p in defined_area_points["relative points"]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True, COLORS[defined_area_name])
    offset_y = 0
    for area_name, area_color in COLORS.items():
        draw_text(f"{area_name}", img, img.shape[1]- 300, 30 + offset_y, area_color)
        offset_y += 30

def draw_text(text,img, x , y, color = (255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (110, 25)
    fontScale = 1
    thickness = 2
    cv2.putText(img, text, (x, y), font, 1, color, thickness, cv2.LINE_AA)

def set_points(img, defined_areas, area_name):
    clone = img.copy()
    height, width , _ = img.shape
    points = []
    coords = []
    cv2.namedWindow("area_definition", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("area_definition", partial(add_point, points, width, height, coords))
    abort = False

    while True:
        #show areas already definited
        draw_areas(img, defined_areas)
        for defined_area_name, defined_area_points in defined_areas.items():
            abs_pts = np.array([[p['x']*width, p['y']*height] for p in defined_area_points["relative points"]], np.int32)
            abs_pts = abs_pts.reshape((-1,1,2))
            cv2.polylines(img,[abs_pts],True, COLORS[defined_area_name])

        # show points of area being definited
        for p in points:
            abs_p = tuple(np.array([p['x']*width,p['y']*height]).astype(np.int32))
            cv2.circle(img, abs_p, 4, COLORS[area_name], -1)

        draw_text(f"Defining {area_name}", img, 100, 20)
        draw_text(f"Press r to reset", img, 100, 50)
        draw_text(f"Press q to quit", img, 100, 70)
        draw_text(f"Press c to complete", img, 100, 90)
        cv2.resizeWindow("area_definition", 1000, 1000)
        cv2.imshow("area_definition", img)

        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            img = clone.copy()
        elif key == ord("c"):
            break
        # if the 'c' key is pressed, break from the loop
        elif key == ord("q"):
            abort = True
            break
    if not abort:
        return points, coords
    if abort:
        cv2.destroyAllWindows()
        raise Exception("Quited Program")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--sample_data_path", help="path to directory with acolite output", 
                        default=os.path.join("sample_data", "2021-01-25"))
    parser.add_argument("-o", "--out_json", help="json for the output results",
                        default="water_mask.json")
    
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    sample_data_path = Path(args.sample_data_path)
    data = process_sentinel2.DayData(sample_data_path)

    output_path = args.out_json

    # if area_definition file doesn't exist, define areas
    if not Path(output_path).exists():
        defined_areas = {}
        points_area, coords = set_points(np.copy(data.rgb), defined_areas, "valid water")
        defined_areas["valid water"] = {"relative points":points_area}
        defined_areas["valid water"]["lon/lat"] = [{"longitude": float(data.longitude[coord["y"], coord["x"]]), 
                                                    "latitude": float(data.latitude[coord["y"], coord["x"]])} 
                                                    for coord in coords]
    else:
        with open(output_path) as json_file:
            defined_areas = json.load(json_file)

    # show result
    cv2.namedWindow("area_definition", cv2.WINDOW_NORMAL)
    draw_areas(data.rgb, defined_areas)
    draw_text("Press s to SAVE", data.rgb, 100, 20)
    cv2.resizeWindow("area_definition", 1000, 1000)
    cv2.imshow("area_definition", data.rgb)
    key = cv2.waitKey(0) & 0xFF
    # save to json if pressing s
    if key == ord("s"):
        with open(output_path, 'w') as outfile:
            json.dump(defined_areas, outfile)
    cv2.destroyAllWindows()
