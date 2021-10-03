import cv2
import argparse
from tqdm import tqdm
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

# preloaded = [# sur
#             (-34.841896,-55.124554),
#             (-34.842670,-55.124363),
#             (-34.843388,-55.124084),
#             (-34.844120,-55.123917),
#             (-34.844803,-55.123581),
#             (-34.845470,-55.123329),
#             (-34.845757,-55.122528),
#             (-34.846046,-55.121731),
#             (-34.846363,-55.120941),
#             (-34.846489,-55.120060),
#             (-34.847328,-55.118622),
#             (-34.848049,-55.118713),
#             (-34.848778,-55.118721),
#             (-34.848412,-55.117973),
#             (-34.848007,-55.117233),
#             (-34.848095,-55.116329),
#             (-34.848179,-55.115429),
#             (-34.848244,-55.114578),
#             (-34.848476,-55.113739),
#             (-34.848526,-55.112934),
#             (-34.848301,-55.112099),
#             (-34.847683,-55.111664),
#             (-34.847027,-55.111271),
#             (-34.846401,-55.110752),
#             (-34.845741,-55.110382),
#             (-34.844952,-55.110188),
#             (-34.844261,-55.110210),
#             (-34.843727,-55.110916),
#             (-34.843151,-55.111576),
#             # norte
#             (-34.839691,-55.115154),
#             (-34.839989,-55.115360),
#             (-34.840191,-55.115730),
#             (-34.840454,-55.116020),
#             (-34.840683,-55.116371),
#             (-34.840878,-55.116745),
#             (-34.841038,-55.117142),
#             (-34.841217,-55.117546),
#             (-34.841190,-55.117992),
#             (-34.841270,-55.118450),
#             (-34.841366,-55.118900),
#             (-34.841412,-55.119339),
#             (-34.841446,-55.119766),
#             (-34.841373,-55.120228),
#             (-34.841351,-55.120655),
#             (-34.841316,-55.121098),
#             (-34.841263,-55.121574),
#             (-34.841267,-55.122021),
#             (-34.841213,-55.122463),
#             (-34.841148,-55.122890),
#             (-34.841045,-55.123295),
#             (-34.840958,-55.123714),
#             (-34.840874,-55.124161),
#             (-34.840706,-55.124542),
#             ]
preloaded = []

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

def coords_to_xy(p_c, day_data):
    # list of {
    #           'x': x,
    #           'y': y
    #         }
    output_coords = []
    # list of {
    #           'x': x / width,
    #           'y': y / height
    #         }
    output_points = []
    h = day_data.rgb.shape[0]
    w = day_data.rgb.shape[1]
    pbar = tqdm(total=len(p_c))
    for coord in p_c:
        y, x = day_data.get_pos_index(coord[0], coord[1])
        output_coords.append({'x': x, 'y': y})
        output_points.append({'x': x/w, 'y': y/h})
        pbar.update(1)
    pbar.close()
    return output_points, output_coords
    

def set_points(img, defined_areas, area_name, preloaded_points=[], preloaded_coords=[]):
    # preloaded_coords must be list of
    clone = img.copy()
    height, width , _ = img.shape
    if preloaded_points == [] and preloaded_coords == []:
        points = []
        coords = []
    else:
        points = preloaded_points
        coords = preloaded_coords

    cv2.namedWindow("area_definition", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("area_definition", partial(add_point, points, width, height, coords))
    abort = False

    while True:
        #show areas already defined
        draw_areas(img, defined_areas)
        for defined_area_name, defined_area_points in defined_areas.items():
            abs_pts = np.array([[p['x']*width, p['y']*height] for p in defined_area_points["relative points"]], np.int32)
            abs_pts = abs_pts.reshape((-1,1,2))
            cv2.polylines(img,[abs_pts],True, COLORS[defined_area_name])

        # show points of area being defined
        for p in points:
            abs_p = tuple(np.array([p['x']*width, p['y']*height]).astype(np.int32))
            cv2.circle(img, abs_p, 2, COLORS[area_name], -1)

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
    
    if preloaded != []:
        preloaded_points, preloaded_coords = coords_to_xy(preloaded, data)
    else:
        preloaded_coords = []
        preloaded_points = []

    output_path = args.out_json

    # if area_definition file doesn't exist, define areas
    if not Path(output_path).exists():
        defined_areas = {}
        points_xy, coords = set_points(np.copy(data.rgb), defined_areas, "valid water", preloaded_points, preloaded_coords)
        defined_areas["valid water"] = {"relative points":points_xy}
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
