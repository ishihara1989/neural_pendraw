# neural_pendraw

[DRAW](https://arxiv.org/abs/1502.04623)は最初全体に大まかに書き込んでから細部を仕上げているような出力が得られる.
ここで目指すのはペンで描いたような画像シークエンスを生成することである.
現在Early stage

## やること

画像の座標に対する微分をできるようにする
ペン座標Onehot*(x,yのグリッド)で座標が出るのでそれでなんとか

ペンの大きさを微分できるようにする
ペンの座標を複数持ってそれぞれで切り替える?
座標出して区分線形関数を低数倍して近似

1を超える出力を0.001倍くらいに圧縮する

LSTM

