import sys
print(sys.version)

sys.path.append('/home/vicroni/Desktop/RNNSLAM/src/RNN')
import RNN_pred_colon_reproj
net = RNN_pred_colon_reproj.RNN_depth_pred('/home/vicroni/Desktop/models/model-145000', output_dir='/home/vicroni/Desktop/')



net.assign_keyframe_by_path('/home/vicroni/Desktop/prueba.jpg')
net.predict('/home/vicroni/Desktop/prueba.jpg')
net.update()