# -*- encoding: utf-8 -*-
# stub: slim 3.0.9 ruby lib

Gem::Specification.new do |s|
  s.name = "slim".freeze
  s.version = "3.0.9"

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Daniel Mendler".freeze, "Andrew Stone".freeze, "Fred Wu".freeze]
  s.date = "2017-11-09"
  s.description = "Slim is a template language whose goal is reduce the syntax to the essential parts without becoming cryptic.".freeze
  s.email = ["mail@daniel-mendler.de".freeze, "andy@stonean.com".freeze, "ifredwu@gmail.com".freeze]
  s.executables = ["slimrb".freeze]
  s.files = ["bin/slimrb".freeze]
  s.homepage = "http://slim-lang.com/".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.0.0".freeze)
  s.rubygems_version = "2.7.6.2".freeze
  s.summary = "Slim is a template language.".freeze

  s.installed_by_version = "2.7.6.2" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<temple>.freeze, ["< 0.9", ">= 0.7.6"])
      s.add_runtime_dependency(%q<tilt>.freeze, ["< 2.1", ">= 1.3.3"])
    else
      s.add_dependency(%q<temple>.freeze, ["< 0.9", ">= 0.7.6"])
      s.add_dependency(%q<tilt>.freeze, ["< 2.1", ">= 1.3.3"])
    end
  else
    s.add_dependency(%q<temple>.freeze, ["< 0.9", ">= 0.7.6"])
    s.add_dependency(%q<tilt>.freeze, ["< 2.1", ">= 1.3.3"])
  end
end
