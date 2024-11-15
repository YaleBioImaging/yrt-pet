function write_gcdata(data, fname, precision, flag_legacy)
%Write binary data from file in YRT-PET format
%
% function write_gcdata(data, fname, flag_legacy)
%
% Inputs
% - data: Matrix to write to disk
% - fname: Output filename
% - precision: Data format (e.g. 'float32', 'float64')
% - flag_legacy (default: false): When true, write binary data without header
%

if nargin < 3
    precision = 'float32';
end
if nargin < 4
    flag_legacy = false;
end

fid = fopen(fname, 'wb');
if fid == -1
    error('Could not open file "%s"', fname);
end
if ~flag_legacy
    magic_number = 732174000;
    fwrite(fid, magic_number, 'int32');
    fwrite(fid, ndims(data), 'int32');
    fwrite(fid, fliplr(size(data)), 'int64');
end
fwrite(fid, data(:), precision);
fclose(fid);

end
