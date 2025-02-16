import sys

def generate_youtube_urls(input_file, output_file):
    try:
        # Read the input file
        with open(input_file, 'r') as file:
            video_ids = file.readlines()
        
        # Filter out "AccessDenie" and empty lines
        filtered_ids = [video_id.strip() for video_id in video_ids if video_id.strip() and video_id.strip() != "AccessDenie"]
        
        # Convert to YouTube URLs
        youtube_urls = [f"http://www.youtube.com/watch?v={video_id}" for video_id in filtered_ids]
        
        # Write the URLs to the output file
        with open(output_file, 'w') as file:
            file.write("\n".join(youtube_urls))
        
        print(f"Processed {len(filtered_ids)} video IDs. URLs saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_url.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    generate_youtube_urls(input_file, output_file)