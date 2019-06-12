# apex_frame_extractor
extractor apex frames from onset-apex-offset videos

夸张帧选取：
step1: 计算每帧相对于初始帧的自定义运动幅度；
step2: 对自定义运算幅度进行10%帧长的平滑（考虑到特征点检测等各种噪音，平滑更准确一点）
step3: 删掉开头和结尾不足够平滑的20%的帧
step4: 剩余的80%的帧种选择自定义运动幅度最大的5帧

自定义运动幅度：参考了文献[1],文献[1]是微笑识别的数据库，所以只考虑了唇角来定义运动幅度。我们的问题涉及多种表情，
基于邻域先验“绝大多数表情只与眼睛、鼻子、嘴唇的区域有关[2]”,我们使用类似[1]的方式定义了左眼、右眼、鼻子、嘴唇
的的运动幅度，然后取其平均作为最终的运动幅度。

以左眼的运动幅度为例：
left_eye_center(第一帧的左眼中心)， 
left_eye_len_base(第一帧的左眼的两个眼角与左眼中心的距离之和）
当前帧的left_eye_len(左眼的两个眼角与初始帧的左眼中心的距离之和）
amplitude(left_eye) = abs(left_eye_len-left_eye_len_base)/left_eye_len_base

[1] Dibeklioğlu H, Salah A A, Gevers T. Recognition of genuine smiles[J]. IEEE Transactions on Multimedia, 2015, 17(3): 279-294.
[2] Zhong L, Liu Q, Yang P, et al. Learning multiscale active facial patches for expression analysis[J]. IEEE transactions on cybernetics, 2015, 45(8): 1499-1510.
