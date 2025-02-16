# This script is inspired by and partially based on the code from the following repository:
# https://github.com/gsssrao/youtube-8m-videos-frames
# Special thanks to the contributors of that repository for their valuable work.

# Check number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: bash downloadcategoryids.sh <number-of-videos-per-category> <category-name>"
    exit 1
fi

js=".js"
txt=".txt"
name="${@:2}"
numVideos=$1
url='https://storage.googleapis.com/data.yt8m.org/2/j/v/'

echo "Number of videos: " $1

if [ "$(uname)" == "Darwin" ]; then
    mid=$(grep -E "\t$name \(" youtube8mcategories.txt | grep -o "\".*\"" | sed -n 's/"\(.*\)"/\1/p')
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    mid=$(grep -P "\t$name \(" youtube8mcategories.txt | grep -o "\".*\"" | sed -n 's/"\(.*\)"/\1/p')
fi

txtName=$mid$txt
mid=$mid$js

output_folder="category-${name}-info"
mkdir -p "$output_folder"

curl -o "$output_folder/$txtName" $url$mid

if [ "$(uname)" == "Darwin" ]; then
    grep -E -oh [a-zA-Z0-9_-]{4} "$output_folder/$txtName" > "$output_folder/tmp$txtName"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    grep -P -oh [a-zA-Z0-9_-]{4} "$output_folder/$txtName" > "$output_folder/tmp$txtName"
fi

tail -n +2 "$output_folder/tmp$txtName" > "$output_folder/$txtName"

if [ "$1" -eq 0 ]; then
    mv "$output_folder/$txtName" "$output_folder/tmp$txtName"
else
    awk -v var="$numVideos" ' NR <= var' "$output_folder/$txtName" > "$output_folder/tmp$txtName"
fi

url1='https://storage.googleapis.com/data.yt8m.org/2/j/i/'

cut -c1-2 "$output_folder/tmp$txtName" > "$output_folder/tmp2$txtName"

rm -rf "$output_folder/$txtName"

exec 6<"$output_folder/tmp2$txtName"
while read -r line
do
    read -r firstTwoChars <&6
    echo "${url1}${firstTwoChars}/${line}.js" >> "$output_folder/Movieclips_video_ids.txt"
done <"$output_folder/tmp${txtName}"
exec 6<&-

rm -rf "$output_folder/tmp$txtName"
while IFS= read -r line
do
    if [ "$(uname)" == "Darwin" ]; then
        curl "$line" | grep -E -oh [a-zA-Z0-9_-]{11} >> "$output_folder/tmp$txtName"
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        curl "$line" | grep -P -oh [a-zA-Z0-9_-]{11} >> "$output_folder/tmp$txtName"
    fi
done < "$output_folder/Movieclips_video_ids.txt"

mv "$output_folder/tmp$txtName" "$output_folder/Movieclips_video_ids.txt"
rm -rf "$output_folder/tmp$txtName"
rm -rf "$output_folder/tmp2$txtName"

echo "Completed downloading YouTube video IDs to $output_folder/Movieclips_video_ids.txt"

# Run create_url.py script
python3 create_url.py "$output_folder/Movieclips_video_ids.txt" "$output_folder/Movieclips_urls.txt"

echo "YouTube URLs have been saved to $output_folder/Movieclips_urls.txt"