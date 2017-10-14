% a script to explore the demo Person Image collection
% Each face is circled and colored according to ground truth gender
% Pink for Female, Blue for Male
%
function viewGroupImages(coll)
    %load matlabData.mat

    for i = 1:1:max(coll.images)
        these       = find(coll.images==i);
        allxyxy     = coll.facePosSize(these,1:4);
        imname      = coll.faceData(these(1)).name;
        classes     = coll.genClass(these);
        guesses     = coll.genClass(these); 
        renderFaces(imname,allxyxy,classes,guesses)
        pause
    end
end

function out = renderFaces(name,allxyxy,classes,guesses,colors)
%function out = renderFaces(name,allxyxy,classes,guesses,colors)
%
% This function will render an image of the faces in the image.
%
%
% name is the image name
% allxyxy   Nx4 the x and y left eye, x and y right eye, 1 row per face
% classes   the classes for each face Nx1  drawn from M values
% colors    the color for each class Mx3
%
if(nargin<5)
    colors =[246 128 182; 100 100 255]./255;  %pink and blue    
end
im = imread(name);

xy = [allxyxy(:,1)+allxyxy(:,3) allxyxy(:,2)+allxyxy(:,4)]/2;
sizes = sqrt((allxyxy(:,1)-allxyxy(:,3)).^2+(allxyxy(:,2)-allxyxy(:,4)).^2);

N = size(allxyxy,1); 

close all
imagesc(im);axis image; axis off
set(gcf,'Position',[1 31 1280 696])
hold on;
ang = 0:.1:6.3;
for i = 1:1:N
    right = classes(i)==guesses(i);
    plot(xy(i,1)+cos(ang)*(2*sizes(i)),xy(i,2)+sin(ang)*(sizes(i)*2),'LineWidth',3,'Color',colors(guesses(i),:) )
    if(~right)
            plot(xy(i,1)+cos(ang)*(2*sizes(i)+3),xy(i,2)+sin(ang)*(sizes(i)*2+3),'--','LineWidth',3,'Color',[1 0 0 ] )

    end
end

end