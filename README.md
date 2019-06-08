# oreoreSGEMM

もともとはレジスターブロッキングの勉強のために始めました。
https://cnugteren.github.io/tutorial/pages/page8.html
ここのOpenCL SGEMM TutorialをCUDAにコピペして遊んだりしながら自分なりに改良してoreoreSGEMMを作りました。結果、このコピペSGEMMより少く速くできました。
さらにそのあとcuBLASのSGEMMと比較してみました。
このサイトで紹介されてるコードはグラフを見る限りcuBLASより2倍遅い感じでしたが、やってみると意外とそこまで差がつかなかったです。
むしろTuring GPUではcuBLASが最適化されてないのかコピペSGEMMのほうがcuBLASより速かったです。私が速くするまでもなかったようです。
なおoreoreSGEMMは行列の行と列がそれぞれ128の倍数でないと機能しません。


結果
■GTX1080(Pascal)　	理論性能	8.88TFLOPS
・コピペSGEMM	実行性能5.69-5.70TFLOPS(実行効率64.18%)
・oreoreSGEMM	実行性能6.51-6.62TFLOPS(実行効率74.17%)
・cuBLAS	実行性能7.15-7.26TFLOPS(実行効率81.8%)

■RTX2080Ti(Turing)	理論性能14.23TFLOPS
・コピペSGEMM	実行性能10.95-11.11TFLOPS(実行効率76.9%)
・oreoreSGEMM	実行性能12.20-12.33TFLOPS(実行効率85.7%)
・cuBLAS	実行性能10.25-10.46TFLOPS(実行効率72.0%)



※row majorとかcol majorとかがこんがらがってて、各SGEMMが行ってることがc=B.T*A , c=A*B , c=B.T*A.T　とぐちゃぐちゃのままです。
