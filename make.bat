ffmpeg -i output/%1/3d_%%d.png -c:v libx264 -preset veryslow -pix_fmt yuv420p -crf 27 -profile:v main output/%1.mp4

