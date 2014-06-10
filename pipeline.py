import getpass, re, json, os, requests, zipfile, csv, sqlite3

comment_re = re.compile(
	'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
	re.DOTALL | re.MULTILINE
)

def processPipeline():
	pipelineFile = "pipeline.json"

	with open(pipelineFile, 'r') as f:
		jsontext = f.read()
		jsontext = json_minify(jsontext)
		pipelines = json.loads(jsontext)

	session = getKaggleSession()

	for pipeline in pipelines:
		if "dest" not in pipeline:
			pipeline["dest"] = pipeline["url"].split("/")[-1]
		pipeline["session"] = session
		executePipeline(pipeline)
		print ""

def getKaggleSession():
	username = passwd = ""
	if os.path.isfile("auth.json"):
		with open("auth.json", "r") as f:
			auth_json = json.load(f)
			x = auth_json["kaggle.com"]
			username = x["un"]
			passwd = x["pw"]
	else:
		print "enter kaggle account:"
		username = raw_input()
		print "enter kaggle password:"
		passwd = getpass.getpass()

	session = requests.Session()
	session.get("https://www.kaggle.com/account/login")
	session.post(
		"https://www.kaggle.com/account/login",
		data={
			"UserName": username,
			"Password": passwd
		}
	)
	return session

def executePipeline(pipeline):
	lastfile = downloadFile(pipeline)
	if lastfile.endswith(".zip"):
		lastfile = unzipFile(pipeline, lastfile)
	# if lastfile.endswith(".csv"):
	# 	lastfile = importCsv(pipeline, lastfile)

def downloadFile(pipeline):
	url = pipeline["url"]
	filename = pipeline["dest"]

	if os.path.isfile(filename):
		print "already downloaded '%s', skipping..." % filename
	else:
		print "downloading '%s'..." % filename

		with open(filename, 'w') as outf:
			#use redirect to log in
			req = pipeline["session"].get(pipeline["url"])

			if req.status_code != 200:
				raise "download request failed"

			for chunk in req.iter_content(chunk_size = 512*1024):
				if chunk:
					outf.write(chunk)
			outf.flush()
	return filename

def unzipFile(pipeline, lastfile):
	with zipfile.ZipFile(lastfile) as zf:
		files = zf.infolist()
		if len(files) > 1:
			raise "Too many files in '%s', need to update pipeline.py to handle this" % lastfile
		unzippedFile = files[0]

		print "unzipping '%s'..." % lastfile
		zf.extractall(".")

		return unzippedFile

def importCsv(pipeline, lastfile):
	pass

# https://github.com/getify/JSON.minify/blob/master/minify_json.py
def json_minify(string, strip_space=True):
	tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
	end_slashes_re = re.compile(r'(\\)*$')

	in_string = False
	in_multi = False
	in_single = False

	new_str = []
	index = 0

	for match in re.finditer(tokenizer, string):

		if not (in_multi or in_single):
			tmp = string[index:match.start()]
			if not in_string and strip_space:
				# replace white space as defined in standard
				tmp = re.sub('[ \t\n\r]+', '', tmp)
			new_str.append(tmp)

		index = match.end()
		val = match.group()

		if val == '"' and not (in_multi or in_single):
			escaped = end_slashes_re.search(string, 0, match.start())

			# start of string or unescaped quote character to end string
			if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
				in_string = not in_string
			index -= 1 # include " character in next catch
		elif not (in_string or in_multi or in_single):
			if val == '/*':
				in_multi = True
			elif val == '//':
				in_single = True
		elif val == '*/' and in_multi and not (in_string or in_single):
			in_multi = False
		elif val in '\r\n' and not (in_multi or in_string) and in_single:
			in_single = False
		elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):
			new_str.append(val)

	new_str.append(string[index:])
	return ''.join(new_str)

if __name__ == "__main__":
	processPipeline()