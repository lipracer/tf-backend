// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>

using namespace clang::tooling;
using namespace llvm;

// CHECK glib abi
static_assert(_GLIBCXX_USE_CXX11_ABI == 1, "");

static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static cl::opt<std::string> output_path("output-path", cl::desc("rcodegen output path]"), cl::value_desc("a full path"),
                                        cl::ValueRequired, cl::NotHidden, cl::cat(MyToolCategory));

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.

using namespace clang;
using namespace clang::ast_matchers;

class CodeGenerator
{
public:
    static CodeGenerator& instance()
    {
        static CodeGenerator generator;
        return generator;
    }

    void writeString(StringRef str) {
        os_ << str;
    }

    ~CodeGenerator()
    {
        os_.flush();
        std::string path = output_path;
        if (path.back() != '/')
        {
            path.push_back('/');
        }
        path += "op_registry.cpp";
        std::ofstream ofs(path);
        if (!ofs.is_open())
        {
            llvm_unreachable("can't open file!");
        }
        ofs << buf_;
        ofs.flush();
        ofs.close();
    }

private:
    CodeGenerator() : os_(buf_) {}

    std::string buf_;
    llvm::raw_string_ostream os_;
};

class RegistryCodegen : public MatchFinder::MatchCallback
{
public:
    void run(const MatchFinder::MatchResult& Result) override
    {
        if (const FunctionDecl* funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("function"))
        {
            const DeclContext* DeclCtx = funcDecl->getDeclContext();

            const auto* Namespace = dyn_cast<NamespaceDecl>(DeclCtx);
            if (!Namespace)
            {
                return;
            }
            if (Namespace->getNameAsString() != "autogen")
            {
                return;
            }

            for (const auto& param : llvm::make_range(funcDecl->param_begin(), funcDecl->param_end()))
            {
                llvm::errs() << "param name:" << param->getName() << "\n";
            }
        }
    }
};

// auto CxxMemberCallExprM = cxxDependentScopeMemberExpr().bind("memberCall");

int main(int argc, const char** argv)
{
    (void)CodeGenerator::instance();
    CodeGenerator::instance().writeString("");
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory, cl::Optional, nullptr);
    if (!ExpectedParser)
    {
        // Fail gracefully for unsupported options.
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    MatchFinder Finder;

    auto funcDeclMatcher = functionDecl().bind("function");

    RegistryCodegen codegen;
    Finder.addMatcher(funcDeclMatcher, &codegen);

    int ExitCode = Tool.run(newFrontendActionFactory(&Finder).get());
    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
    TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
                                  &DiagnosticPrinter, false);

    return ExitCode;
}
