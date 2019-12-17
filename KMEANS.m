clc 
clear

%Loading image and converting to double
image_uint8  = imread('italy.jpg');
%image_uint8  = imread('Lenna.tif');
image = double(image_uint8);
[M, N] = size(image(:,:,1));
totalPixels = M * N;

%setting up initial variables
k = 2;
ITERATIONS = 100000000;
labels     = zeros(size(image(:,:,1)));
epsilon    = 0.00001;
cost       = 0;
old_cost   = 0;
new_cost   = 0;

%creating random centroids each row is a centroid with RGB values
centroids = randi(255, k, 3); 

tic
%start of the KMEANS loop
for I = 1:ITERATIONS
    
    %saving old centroids for convergence testing
    old_centroids = centroids;
    
    %Assigning each pixel a centroid (stored in label matrix)
    for r = 1:M
        for c = 1:N 
            %calculate the distance between current pixel and each centroid
            %gets the RGB of pixel at (r,c,:) and puts it into a vector
            pixel = permute(image(r,c,:), [3 2 1])';
            %next three lines calculate the distance between each centroid
            %and current pixel
            distance =  ((centroids - pixel).^2);%(RGB_k - RGB_pixel)^2
            distance =  sum(distance, 2);%(delta_r + delta_g + delta_b)
            distance =  sqrt(distance);
            %stores the centroid number it belongs to in label matrix
            [~,labels(r,c)] = min(distance);
            %adds cost(min distance) for current pixel
            cost = cost + min(distance);
        end
    end
    
    %store cost and clear for next value
    old_cost = new_cost;
    new_cost = cost;
    cost_vector(I) = cost;
    cost = 0;
    
    %Moving centroids based on the average RGB value of each group
    new_centroids = zeros(k,3);
    for i = 1:k %goes through each group K
        %Select the pixels to compute avereage
        group = double((labels==i));
        number_pixels = sum(sum(group));
        pixels = group .* image;
        
        if (number_pixels == 0)% if no pixel in group reassign random centroid
            new_centroids(i,:) = randi(255,1,3);
        else % move centroid to the average RGB value of the group
            avg = sum(sum(pixels,1),2) / number_pixels;
            avg = permute(avg, [3 2 1])';
            new_centroids(i,:) = avg;
        end
    end
    
    centroids = new_centroids;
    
    %terminate if the difference in error meets threshold (epsilon)
    err = abs(new_cost - old_cost);
    if err <= epsilon
        break;
    end
end
toc

new_image = zeros(size(image));
for i = 1:k
    %mask for targeted group k
    group = double((labels == i));
    
    %gets new RGB value from centroid k
    avg_rgb = permute(centroids(i,:), [3 1 2]); 
    
    %add the avg rgb value to new image
    new_pixels = group .* avg_rgb;
    new_image = new_image + new_pixels;
end

%saving image as unsigned int 8
image_new = uint8(new_image);
imwrite(image_new, 'TESTING.tif');

%plotting cost
figure
plot(1:1:I, cost_vector)
xlabel('Iteration');
ylabel('Cost');
title('Cost function for Italy k : 2');



