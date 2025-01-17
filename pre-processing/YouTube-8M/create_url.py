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

# Example usage
input_file = "/Users/imch/workspace/youtube-8m-videos-frames/category-ids/Movieclips_video_id.txt"  # Replace with your input file name
output_file = "Movieclips_urls.txt"  # Replace with your desired output file name
generate_youtube_urls(input_file, output_file)