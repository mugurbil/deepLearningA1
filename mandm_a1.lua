-- Modification of Clement Farabet's 1_data.lua
-- Deep Learning A1
-- 02/09/2015
-- M&M: Mehmet Ugurbil, Mark Liu

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'math'

----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files.

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'

test_file = 'test_32x32.t7'

if not paths.filep(test_file) then
    os.execute('wget ' .. www .. test_file)
end

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

-- Torch's slicing syntax can be a little bit frightening. I've
-- provided a little tutorial on this, in this same directory:
-- A_slicing.lua

-- Finally we load the test data.

loaded = torch.load(test_file,'ascii')
testData = {
    data = loaded.X:transpose(3,4),
        labels = loaded.y[1]
}

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
                                                           -- data is stored on bytes). Types can be easily converted in Torch,
-- in general by doing: dst = src:type('torch.TypeTensor'),
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'

for i = 1,testData.data:size()[1] do
    testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'

-- I saved these values from a previous run of the training data so we don't have to do it every time
-- These numbers come from the small data size
mean = {111.85664054059, 98.071901586681, 103.83652386792}
std = {50.467668872399, 60.806183688379, 62.810269594025}

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
    -- normalize each channel globally:
        testData.data[{ {},i,{},{} }]:add(-mean[i])
            testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module,
                                            -- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
    for i = 1, testData.data:size()[1] do
        testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
            end
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
    
    testMean = testData.data[{ {},i }]:mean()
        testStd = testData.data[{ {},i }]:std()
            
            print('test data, '..channel..'-channel, mean: ' .. testMean)
                print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

print '==> generating predictions'

file = io.open('predictions.csv', 'w')
file:write('Id,Prediction\n')

model = torch.load('model_ascii.net', 'ascii')
model:float()
criterion = nn.ClassNLLCriterion()

function fix_label(label)
    if label < 10 then label=label else label=0 end
    return label
end

-- The prediction is the one that gives the lowest error
for i = 1, testData.data:size()[1] do
    if i % 50 == 0 then print("Prediction #"..i) end
    local best_label = 0
    local best_score = math.huge
    for label = 1, 10 do
        local output = model:forward(testData.data[i])
        local score = criterion:forward(output, label)
        if score < best_score then
            best_score = score
                best_label = fix_label(label)
            end
end
    file:write(i,',',best_label,'\n')
    best_score = math.huge
end

file:close()




