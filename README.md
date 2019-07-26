# wikiart

[wikiart](https://www.wikiart.org/)の絵画画像を分類する。
絵画画像データは、[Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers/data)から取得した。

実験では、[知っておきたい世界の有名画家40人と代表作を徹底解説](https://media.thisisgallery.com/20185441)を参考に、著名な作家40人を対象とした。

## 実験
前処理は、画像のクロッピングだけ行い、簡単なCNNのモデルで分類を行なった。
テストデータに対して、acc: 0.4程度だった。 
2019/7/22

Test loss: 1.833
Test accuracy: 0.468
2019/7/23


## pretrained CNNを用いて抽出した特徴量の可視化
![UMAP](https://github.com/mrkmakr/wikiart/blob/master/src/res18/2d.jpg "UMAP")
