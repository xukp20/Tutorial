### Pix2Text

recognize:

- mfd: for formula recognization, recogize_by_mfd
  - LayoutAnalyzer使用cnstd中的模型
    - analyze: 版面分析 img -> list of (type, box, score)
      - yolov7 model output
      - confidence + box + category
    - 仅识别出公式
  - get patch -> latex_model
    - LatexOCR: checkpoints in formula
      - encoder: 输出为decoder输入的context
      - decoder:输入bos * batch_size，输入context
  - mask formula with 255
  - genernal OCR for general texts, CnOcr Type
    - 如果识别到的box中含有embedding公式，将图片patch按embedding分为多个text部分，分别识别
    - 使用det_model的detect，只检测
  - 使用ocr_for_single_line识别text
    - 使用rec_model的recognize