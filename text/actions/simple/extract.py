def extract_before_colon(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as input_file, open(output_filename, 'w', encoding='utf-8') as output_file:

            for line in input_file:

                content_before_colon = line.split(':')[0].strip()
                # if '(' in line:
                #     content_before_colon = line.split('(')[0].strip()

                output_file.write(content_before_colon + '\n')

        print(f"Extraction completed. Results saved in {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 调用函数并指定输入和输出文件名
extract_before_colon("text/detail/PKU-view.txt", "text/simple/PKU-view.txt")