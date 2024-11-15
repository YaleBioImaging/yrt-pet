function data = read_gcdata(fname, precision, flag_legacy, dims)
%Read binary data from file in YRT-PET format
%
% function read_gcdata(fname, precision, flag_legacy, dims)
%
% Inputs
% - fname: Output filename
% - precision: Data format (e.g. 'float32', 'float64')
% - flag_legacy (default: false): When true, read binary data without header
% - dims: When in legacy mode, the dimension of the array.
%
% Output
% - data: Matrix read from disk.
%

if nargin < 2
    precision = 'float32';
end
if nargin < 3
    flag_legacy = false;
end
if nargin < 4
    dims = [];
end

fid = fopen(fname, 'rb');
if fid == -1
    error('Could not open file "%s"', fname);
end
if ~flag_legacy
    magic_number = 732174000;
    m0 = fread(fid, 1, 'int32');
    if m0 ~= magic_number
        error('Wrong file format, is file in legacy raw format?');
    end
    num_dims = fread(fid, 1, 'int32');
    dims = fread(fid, num_dims, 'int64');
    if num_dims == 1
        dims = [dims, 1];
    end
    dims = dims(end:-1:1);
end
data_raw = fread(fid, precision);
fclose(fid);

if ~isempty(dims)
    idx_auto = find(dims == -1);
    if length(idx_auto) == 1
        dims(idx_auto) = numel(data_raw) / ...
            prod(dims([1:idx_auto - 1, idx_auto + 1:end]));
    end
    data = reshape(data_raw, dims(:)');
else
    data = data_raw;
end

end
