ffmpeg -i output/%1/3d_%%d.png -c:v libx264 -preset slow -pix_fmt yuv420p -crf 26 output/%1.mp4

