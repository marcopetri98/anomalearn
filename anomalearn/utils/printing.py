import warnings
from datetime import datetime

import colorama
from colorama import Fore, Style


def print_header(heading: str, separator: str = "=", **kwargs) -> None:
	"""Prints a heading.

	Parameters
	----------
	heading: str
		The text to be printed as heading.

	separator: str, default="="
		The separator to be used to limit the heading.

	kwargs
		Keyword arguments. The only accepted is end which is propagated to the
		print function.

	Returns
	-------
	None
	
	.. deprecated:: 0.0.1
		This function is deprecated and will be replaced in favor of built-in
		python logging method since it is pre-built.
	"""
	warnings.warn("This function will be substituted by usage of logging", DeprecationWarning)
	# Checks assumption
	if len(separator) != 1:
		raise ValueError("The separator must be a single character")

	string_to_print = "=" * (len(heading) + 4)
	string_to_print = string_to_print + "\n"
	string_to_print = string_to_print + "= " + heading + " =\n"
	string_to_print = string_to_print + "=" * (len(heading) + 4)

	if "end" in kwargs.keys():
		print(string_to_print, end=kwargs["end"])
	else:
		print(string_to_print)


def print_step(*args, **kwargs) -> None:
	"""Prints a step of execution.

	Parameters
	----------
	args
		Objects representing the message to be printed.

	kwargs
		Keyword arguments. The only accepted is end which is propagated to the
		print function.

	Returns
	-------
	None
	
	.. deprecated:: 0.0.1
		This function is deprecated and will be replaced in favor of built-in
		python logging method since it is pre-built.
	"""
	warnings.warn("This function will be substituted by usage of logging", DeprecationWarning)
	if len(args) == 0:
		raise ValueError("An execution step print must have a text")

	current_timestamp = datetime.now()
	step_print = "[" + current_timestamp.strftime("%Y/%m/%d %H:%M:%S") + "] "
	step_print += "Execution: "
	if len(args) != 0:
		for var in args:
			if isinstance(var, str):
				step_print += var
			else:
				step_print += str(var)

	if "end" in kwargs.keys():
		print(step_print, end=kwargs["end"])
	else:
		print(step_print)


def print_warning(*args, **kwargs) -> None:
	"""Prints a warning on screen.

	Parameters
	----------
	args
		Objects representing the message to be printed.

	kwargs
		Keyword arguments. The only accepted is end which is propagated to the
		print function.

	Returns
	-------
	None
	
	.. deprecated:: 0.0.1
		This function is deprecated and will be replaced in favor of built-in
		python logging method since it is pre-built.
	"""
	warnings.warn("This function will be substituted by usage of logging", DeprecationWarning)
	if len(args) == 0:
		raise ValueError("The warning must have a text")

	colorama.init()
	current_timestamp = datetime.now()
	warning = "[" + current_timestamp.strftime("%Y/%m/%d %H:%M:%S") + "] "
	warning += "WARNING: "
	if len(args) != 0:
		for var in args:
			if isinstance(var, str):
				warning += var
			else:
				warning += str(var)

	if "end" in kwargs.keys():
		print(Fore.RED + Style.BRIGHT + warning + Style.RESET_ALL, end=kwargs["end"])
	else:
		print(Fore.RED + Style.BRIGHT + warning + Style.RESET_ALL)
