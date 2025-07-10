import requests
import base64
import cv2
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import re

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_waypoints(image_rgb, task_instruction="Locate the green object."):
    # import IPython; IPython.embed()
    # assert image_rgb.shape == (480, 640, 3) # may not be 480x640 eg google robot 512x640
    
    if "Select grasp point" in task_instruction:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image> You are currently a robot performing robotic manipulation tasks. Your task instruction: {task_instruction}. Observe the image, use 2D points to mark the manipulated object-centric waypoints to guide the robot to manipulate the object.Typically, the waypoints consists of an ordered sequence of eight 2D points. The format is <point>[[x1, y1], [x2, y2], ...]</point>.###Assistant:"
    elif "Select place point" in task_instruction:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image> You are currently a robot performing robotic manipulation tasks. Your task instruction: {task_instruction}. Observe the image, use 2D points and bounding box to mark the target location where the manipulated object will be moved. In your answer, use <box>[[x1, y1, x2, y2]]</box> to present the bounding box of the target region, and use <point>[[x1, y1], [x2, y2], ...]</point> to mark the points of the free space.###Assistant:"
    else:
        assert False, f"Unknown task instruction: {task_instruction}"
    
    response = requests.post("http://127.0.0.1:40000/worker_generate_stream",
        headers={'User-Agent': 'Client'}, 
        json={
            "model": 'llava-fsd-0506',
            "prompt": prompt,
            "temperature": 1.0,
            "top_p": 0.7,
            "max_new_tokens": 512,
            "stop": '</s>',
            "images": [
                base64.b64encode(cv2.imencode('.png', cv2.cvtColor(cv2.resize(image_rgb, (336, 336)), cv2.COLOR_RGB2BGR))[1]).decode()
            ],
        }, stream=True, timeout=120)

    latest_data = None
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                latest_data = data
            else:
                raise Exception(f"Error from Server: {data['error_code']}")

    print(latest_data["text"])

    match = re.search(r"<[Aa]nswer>.*?<[Pp]oint>(.*?)</[Pp]oint>.*?</[Aa]nswer>", latest_data["text"], re.DOTALL)
    if not match:
        raise Exception("No answer found")
    answer = match.group(1)
    points = np.array(json.loads(answer), dtype=np.float32)
    image_h, image_w = image_rgb.shape[:2]
    points[:, 0] *= image_w / 1000
    points[:, 1] *= image_h / 1000
    
    if "Select grasp point" in task_instruction:
        center_point = points[0]
    else:
        center_point = points.mean(axis=0)

    return center_point, points

if __name__ == "__main__":
    image = cv2.imread("image.png")
    image = cv2.resize(image, (640, 480))
    center_point, points = get_waypoints(image, "Select grasp point of: Locate the coke can.")
    
    # plot points
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    cv2.imwrite("image_with_points.png", image)
    
    print(points)
    print(center_point)