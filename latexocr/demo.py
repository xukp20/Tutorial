from pix2text import Pix2Text, merge_line_texts

img_fp = './imgs/2.jpg'
res_img = img_fp.replace('imgs', 'res').replace('.jpg', '_res.jpg')
import os 
if not os.path.exists('res'):
    os.mkdir('res')

p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
outs = p2t(img_fp, resized_shape=600, save_analysis_res=res_img)
print(outs)
# 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
only_text = merge_line_texts(outs, auto_line_break=True)
print(only_text)