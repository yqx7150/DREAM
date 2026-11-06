% set up matlab path 

% look for directory where this file setup.m is installed
if ~exist('KERdir', 'var')
	KERdir = which('setup');
	KERdir = fileparts(KERdir);
end

if KERdir(end) ~= filesep % make sure there is a '/' at end of directory
	KERdir = [KERdir filesep];
end

path([KERdir 'demo'], path);       % demo
path([KERdir 'eml'], path);        % emission image reconstruction
path([KERdir 'utils'], path);      % utils
path([KERdir 'projector'], path);  % utils
path([KERdir 'work'], path);       % work: my own work folder
path([KERdir 'KMLAA'], path);      % kmlaa
