import requests
import base64
import cv2
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_waypoints(image_rgb, task_instruction="Locate the green object."):
    # import IPython; IPython.embed()
    # assert image_rgb.shape == (480, 640, 3) # may not be 480x640 eg google robot 512x640
    
    task_instruction += " Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{task_instruction}\nASSISTANT:"
    
    response = requests.post("http://127.0.0.1:20000/worker_generate_stream",
        headers={'User-Agent': 'Client'}, 
        json={
            "model": 'robopoint-v1-vicuna-v1.5-13b',
            "prompt": prompt,
            "temperature": 1.0,
            "top_p": 0.7,
            "max_new_tokens": 512,
            "stop": '</s>',
            "images": [
                base64.b64encode(cv2.imencode('.png', cv2.cvtColor(cv2.resize(image_rgb, (640, 480)), cv2.COLOR_RGB2BGR))[1]).decode()
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

    points = np.array(eval(latest_data["text"][len(prompt):]))

    image_h, image_w = image_rgb.shape[:2]
    points[:, 0] *= image_w
    points[:, 1] *= image_h
    
    center_point = points.mean(axis=0)

    return center_point, points

if __name__ == "__main__":
    image = cv2.imread("image.png")
    image = cv2.resize(image, (640, 480))
    center_point, points = get_waypoints(image)
    
    # plot points
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    cv2.imwrite("image_with_points.png", image)
    
    print(points)
    print(center_point)